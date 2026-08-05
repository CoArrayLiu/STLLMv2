"""Download a public Seafile directory into a local folder.

The default arguments download the UniST ``data_release`` share into
``./dataset``. Downloads are written to ``*.part`` files first and resumed on
the next run when possible.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen


DEFAULT_SHARE_URL = (
    "https://cloud.tsinghua.edu.cn/d/87f5954d4f6f4ebd9d70/"
    "?p=%2Fdata_release&mode=list"
)
USER_AGENT = "SeafileDatasetDownloader/1.0"


def parse_share_url(share_url: str) -> tuple[str, str, str]:
    """Return (server origin, share token, remote directory)."""
    parsed = urlparse(share_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"不是有效的 HTTP(S) 链接: {share_url}")

    parts = [part for part in parsed.path.split("/") if part]
    try:
        d_index = parts.index("d")
        token = parts[d_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("链接中未找到 Seafile 分享 token（应包含 /d/<token>/）") from exc

    remote_dir = parse_qs(parsed.query).get("p", ["/"])[0]
    remote_dir = "/" + remote_dir.strip("/") if remote_dir.strip("/") else "/"
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin, token, remote_dir


def request_json(url: str, timeout: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def list_files(
    origin: str, token: str, remote_dir: str, timeout: int
) -> list[dict[str, Any]]:
    """Recursively list files in a public Seafile directory."""
    api_url = (
        f"{origin}/api/v2.1/share-links/{quote(token, safe='')}/dirents/?"
        + urlencode({"path": remote_dir})
    )
    payload = request_json(api_url, timeout)
    entries = payload.get("dirent_list")
    if not isinstance(entries, list):
        raise RuntimeError(f"目录 API 返回了无法识别的数据: {payload!r}")

    files: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("is_dir"):
            child_path = entry.get("folder_path") or entry.get("file_path")
            if not child_path:
                child_name = entry.get("folder_name") or entry.get("file_name")
                child_path = posixpath.join(remote_dir, child_name)
            files.extend(list_files(origin, token, child_path, timeout))
        else:
            file_path = entry.get("file_path")
            if not file_path:
                raise RuntimeError(f"文件条目缺少 file_path: {entry!r}")
            files.append(entry)
    return files


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def local_path_for(remote_path: str, remote_root: str, output_dir: Path) -> Path:
    relative = posixpath.relpath(remote_path, remote_root)
    if relative == ".." or relative.startswith("../") or relative.startswith("/"):
        raise RuntimeError(f"服务端返回了目录范围外的路径: {remote_path}")

    target = output_dir.joinpath(*relative.split("/"))
    output_resolved = output_dir.resolve()
    target_resolved = target.resolve()
    if target_resolved != output_resolved and output_resolved not in target_resolved.parents:
        raise RuntimeError(f"不安全的文件路径: {remote_path}")
    return target


def download_one(
    origin: str,
    token: str,
    remote_path: str,
    destination: Path,
    expected_size: int,
    retries: int,
    timeout: int,
    chunk_size: int,
) -> str:
    """Download one file, resuming a .part file when the server supports Range."""
    if destination.is_file() and destination.stat().st_size == expected_size:
        print(f"[跳过] {destination}（已完整下载）")
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()

    download_url = (
        f"{origin}/d/{quote(token, safe='')}/files/?"
        + urlencode({"p": remote_path, "dl": "1"})
    )

    for attempt in range(1, retries + 2):
        downloaded = partial.stat().st_size if partial.exists() else 0
        if downloaded == expected_size:
            os.replace(partial, destination)
            print(f"[完成] {destination} ({format_bytes(expected_size)})")
            return "downloaded"

        headers = {"User-Agent": USER_AGENT}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        request = Request(download_url, headers=headers)

        try:
            with urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                # A server that ignores Range returns 200. Restart instead of
                # appending a second full copy to the partial file.
                if downloaded and status != 206:
                    downloaded = 0
                    mode = "wb"
                else:
                    mode = "ab" if downloaded else "wb"

                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        percent = downloaded * 100 / expected_size if expected_size else 0
                        print(
                            f"\r[下载] {destination.name}: "
                            f"{format_bytes(downloaded)} / {format_bytes(expected_size)} "
                            f"({percent:5.1f}%)",
                            end="",
                            flush=True,
                        )
            print()

            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise OSError(
                    f"文件大小不匹配: 期望 {expected_size} 字节，实际 {actual_size} 字节"
                )

            os.replace(partial, destination)
            print(f"[完成] {destination}")
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print()
            if attempt > retries:
                raise RuntimeError(
                    f"下载 {remote_path} 失败（已尝试 {attempt} 次）: {exc}"
                ) from exc
            delay = min(2 ** (attempt - 1), 30)
            print(f"[重试] {remote_path}: {exc}；{delay} 秒后继续")
            time.sleep(delay)

    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把公开的 Seafile 目录下载到本地（支持断点续传）。"
    )
    parser.add_argument("--url", default=DEFAULT_SHARE_URL, help="Seafile 公开分享链接")
    parser.add_argument("--output", type=Path, default=Path("dataset"), help="输出目录")
    parser.add_argument("--retries", type=int, default=5, help="每个文件的重试次数")
    parser.add_argument("--timeout", type=int, default=60, help="网络超时秒数")
    parser.add_argument(
        "--chunk-size-mb", type=int, default=4, help="流式下载的分块大小（MiB）"
    )
    parser.add_argument("--list-only", action="store_true", help="只列出文件，不下载")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.retries < 0 or args.timeout <= 0 or args.chunk_size_mb <= 0:
        print("错误：retries 不能为负，timeout 和 chunk-size-mb 必须大于 0", file=sys.stderr)
        return 2

    try:
        origin, token, remote_root = parse_share_url(args.url)
        print(f"正在读取远程目录: {remote_root}")
        files = list_files(origin, token, remote_root, args.timeout)
        total_size = sum(int(item.get("size", 0)) for item in files)
        print(f"共 {len(files)} 个文件，合计 {format_bytes(total_size)}")

        for item in files:
            remote_path = str(item["file_path"])
            size = int(item.get("size", 0))
            destination = local_path_for(remote_path, remote_root, args.output)
            print(f"  {remote_path} -> {destination} ({format_bytes(size)})")

        if args.list_only:
            return 0

        args.output.mkdir(parents=True, exist_ok=True)
        downloaded_count = 0
        skipped_count = 0
        for item in files:
            result = download_one(
                origin=origin,
                token=token,
                remote_path=str(item["file_path"]),
                destination=local_path_for(
                    str(item["file_path"]), remote_root, args.output
                ),
                expected_size=int(item.get("size", 0)),
                retries=args.retries,
                timeout=args.timeout,
                chunk_size=args.chunk_size_mb * 1024 * 1024,
            )
            downloaded_count += result == "downloaded"
            skipped_count += result == "skipped"

        print(f"全部完成：新下载 {downloaded_count} 个，跳过 {skipped_count} 个。")
        return 0
    except KeyboardInterrupt:
        print("\n已中断；再次运行会从 .part 文件继续下载。", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, HTTPError, URLError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())d