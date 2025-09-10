import asyncio
from typing import List
from pathlib import Path
import sys

from PicImageSearch import BaiDu, Network
from PicImageSearch.model import BaiDuResponse


async def _search_image_urls_async(
    image_path: Path, max_results: int = 10
) -> List[str]:
    async with Network() as client:
        baidu = BaiDu(client=client)
        resp: BaiDuResponse = await baidu.search(file=image_path)

    urls: List[str] = []

    if resp.exact_matches:
        for item in resp.exact_matches:
            if item.url:
                urls.append(item.url)

    if len(urls) < max_results and getattr(resp, "raw", None):
        for item in resp.raw:
            if item.url:
                urls.append(item.url)
            if len(urls) >= max_results:
                break
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    return unique_urls[:max_results] if max_results > 0 else unique_urls


def search_image_urls(image_path: Path, max_results: int = 10) -> List[str]:
    return asyncio.run(_search_image_urls_async(image_path, max_results=max_results))


def main() -> None:
    if len(sys.argv) < 2:
        print("python -m tools.baidu_image_search <image_path> [max_results]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        sys.exit(2)

    try:
        max_results = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
    except ValueError:
        max_results = 10

    urls = search_image_urls(file_path, max_results=max_results)
    for u in urls:
        print(u)


if __name__ == "__main__":
    main()
