import asyncio
from datetime import datetime
from pathlib import Path

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


class Size:
    """
    The size of the generated image.
     - gpt-image-1 supports: '1024x1024', '1024x1536', '1536x1024', 'auto'
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1536'
    width_rectangle: str = '1536x1024'
    auto: str = 'auto'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - gpt-image-1 supports: 'low', 'medium', 'high', 'auto'
     - 'high' creates images with finer details and greater consistency across the image.
    """
    low: str = "low"
    medium: str = "medium"
    high: str = "high"
    auto: str = "auto"


async def _save_images(attachments: list[Attachment]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Create DIAL bucket client
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # 2. Iterate through images from attachments, download and save them
        for i, attachment in enumerate(attachments):
            if not attachment.url:
                print(f"⚠️ Attachment {i} has no URL, skipping.")
                continue

            print(f"\n⬇️  Downloading image from: {attachment.url}")
            image_bytes = await bucket_client.get_file(url=attachment.url)

            # 3. Save the image with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tti_generated_{timestamp}_{i}.png"
            output_path = RESULTS_DIR / filename
            output_path.write_bytes(image_bytes)

            print(f"✅ Image saved to: {output_path}")


def start() -> None:
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-image-1-mini-2025-10-06",
        api_key=API_KEY,
    )

    prompt = "Sunny day on Bali: lush tropical rice terraces, warm golden light, blue sky with a few clouds."
    message = Message(
        role=Role.USER,
        content=prompt,
    )

    custom_fields = {
        "size": Size.width_rectangle,
        "quality": Quality.high,
    }

    print(f"\n🎨 Generating image with prompt:\n  \"{prompt}\"")
    response = client.get_completion(
        messages=[message],
        custom_fields=custom_fields,
    )

    if response.custom_content and response.custom_content.attachments:
        attachments = response.custom_content.attachments
        print(f"\n📎 Received {len(attachments)} attachment(s).")
        asyncio.run(_save_images(attachments))
    else:
        print("\n⚠️ No attachments found in the response.")
        print(f"Response content: {response.content}")


start()
