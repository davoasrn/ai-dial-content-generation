import asyncio
import mimetypes
from datetime import datetime
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


async def _put_images(file_names: list[str]) -> list[Attachment]:
    project_root = Path(__file__).parent.parent.parent
    attachments = []

    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        for file_name in file_names:
            image_path = project_root / file_name
            mime_type, _ = mimetypes.guess_type(str(image_path))
            mime_type = mime_type or 'application/octet-stream'

            with open(image_path, 'rb') as f:
                image_bytes = BytesIO(f.read())

            result = await bucket_client.put_file(
                name=file_name,
                mime_type=mime_type,
                content=image_bytes,
            )
            print(f"\n📤 Uploaded '{file_name}': {result}")

            url = result.get("url") or f"files/{result.get('path', file_name)}"
            attachments.append(Attachment(title=file_name, url=url, type=mime_type))

    return attachments


def start() -> None:
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY,
    )

    file_names = ['image-test-1.jpg', 'image-test-2.jpeg']
    attachments = asyncio.run(_put_images(file_names))

    results = []
    for attachment in attachments:
        print(f"\n📎 Attachment: {attachment}")

        message = Message(
            role=Role.USER,
            content="What do you see on this picture?",
            custom_content=CustomContent(attachments=[attachment]),
        )

        response = client.get_completion(messages=[message])
        print(f"\n🤖 Model response:\n{response.content}")
        results.append((attachment.title, response.content))

    # Save results to markdown file
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = RESULTS_DIR / f"dial_itt_results_{timestamp}.md"

    lines = [
        "# DIAL Image-to-Text Results\n",
        f"**Model:** gpt-4o  \n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---\n",
    ]
    for i, (title, content) in enumerate(results, 1):
        lines.append(f"## Image {i}: `{title}`\n")
        lines.append(f"{content}\n")
        lines.append("---\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Results saved to: {md_path}")


start()
