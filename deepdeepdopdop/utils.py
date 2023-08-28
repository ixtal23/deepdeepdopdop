import logging as log

import mimetypes
import urllib

from pathlib import Path

def is_image(path : Path) -> bool:
    if path and path.is_file():
        mimetype, _ = mimetypes.guess_type(path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False

def is_video(path: Path) -> bool:
    if path and path.is_file():
        mimetype, _ = mimetypes.guess_type(path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False

def download(url : str, local_file_path : Path) -> None:
    if local_file_path.exists():
        log.info(f'{url} is already downloaded to {local_file_path}')
        return

    log.info(f'Download from {url} to {local_file_path}')

    directory_path = local_file_path.parent
    if not directory_path.exists():
        directory_path.mkdir(parents = True, exist_ok = True)

    request = urllib.request.urlopen(url)

    content_ength = int(request.headers.get('Content-Length', 0))

    with tqdm(desc = 'Downloading', total = content_ength, unit = 'KB', unit_scale = True, unit_divisor = 1024) as progress:
        urllib.request.urlretrieve(url, local_file_path, reporthook = lambda count, block_size, total_size: progress.update(block_size))
