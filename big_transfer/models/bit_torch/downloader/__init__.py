"""Multithreaded Downloader"""

import click
from .downloader import Downloader, DownloadHandler
from .utils import format_bytes, parse_headers
import sys, os
from typing import List


def download(url: str, destination: str):

    # header =
    # A normal header for downloading

    headers = [
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Encoding: gzip, deflate, br, *",
        "Accept-Language: en-US,en;q=0.9",
        "Connection: keep-alive",
        "DNT: 1",
        "Referer: https://www.github.com/",
        "Upgrade-Insecure-Requests: 1",
        "Cache-Control: max-age=0",
    ]
    header = parse_headers(headers)

    download_handler = DownloadHandler(url, 8, None, True, headers=header)

    if os.path.exists(destination):
        return

     # create dir in path to dest
    dir_path = os.path.join(* destination.split("/")[:-1])
    os.makedirs(dir_path, exist_ok=True)

    download_handler.set_filename(os.path.join(destination))

    #     makedir the path if not exists
#     os.makedirs(destination, exist_ok=True)

    download_handler.start()


# @click.command()
# @click.argument("url", type=str)
# @click.option("--csize", "-s", type=int, help="Chunk size to use, defaults to size/#chunks")
# @click.option("--ccount", "-c", default=8, type=int, help="Number of Chunks to download concurrently")
# @click.option("--output-path", "-o", type=str, help="Path to write the downloaded output file")
# @click.option("--quiet", "-q", is_flag=True, default=False, help="Disable verbose")
# @click.option("--force", "-f", is_flag=True, default=False, help="Suppress confirmation for filename")
# @click.option("--header", "-H", multiple=True, default=[], help="Pass each request header (as in curl)")
# def main(url: str,
#          ccount: int,
#          csize: int,
#          output_path: int,
#          quiet: bool,
#          force: bool,
#          header: List[str]):
#     """Multithreaded Downloader for concurrent downloads"""
#     # parse headers
#     try:
#         headers = utils.parse_headers(header)
#     except Exception as e:
#         click.echo(e, err=True)
#         sys.exit(-1)
#     verbose = not quiet
#     # initialize downloader
#     download_handler = downloader.DownloadHandler(
#          url,
#          ccount,
#          csize,
#          verbose,
#          headers
#     )

#     # confirm if non-parallel download is fine or not
#     if not download_handler.is_parallel():
#          click.secho(
#               "Multithreaded download is not supported!",
#               fg="red"
#          )
#          allow = click.confirm(
#               "Do you want a single threaded download?",
#                default=True,
#                abort=True
#          )

#     # display and confirm name
#     filename = download_handler.get_filename()
#     if output_path is not None:
#         filename = output_path
#         download_handler.set_filename(filename)
#         force = True
#     click.echo("File will be saved as " + click.style(filename, bold=True))
#     if not force:
#         change = click.confirm(
#              "Do you want to change the name?",
#              default=False
#         )
#         if change:
#              filename = click.prompt("New filename")
#              download_handler.set_filename(filename)

#      # display size of file
#     if verbose:
#          size, unit = utils.format_bytes(download_handler.get_size())
#          click.echo(f"Fetching {size:.2f} {unit}s in {ccount} chunks")

#     # start download
#     download_handler.start()
#     click.secho(f"\rSuccess: {filename} downloaded", fg="green")

# if __name__ == '__main__':
#      main()
