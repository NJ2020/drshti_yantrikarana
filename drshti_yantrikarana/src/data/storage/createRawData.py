"""
Usage: python createRawData --dataPath='yourPathToData'

About: Script to create
    - a new folder called raw_data in data directory
    - copy all data folders form given source dir to project data dir

Author: Satish Jasthi
"""
import logging
import sys
import argparse
from pathlib import Path
from distutils.dir_util import copy_tree

from drshti_yantrikarana.config import external_data_dir

sys.path.append(Path(__file__).resolve().parent.parent.parent.parent.parent.as_posix())
from drshti_yantrikarana import data_dir

logging.basicConfig(level=logging.INFO)

argparser = argparse.ArgumentParser(description="Copy raw data to project data dir....................................")
argparser.add_argument('--dataPath',
                       type=str,
                       default=external_data_dir,
                       help="path to your raw data outside project dir")

args = argparser.parse_args()
source_dir = Path(args.dataPath)
projectRawDir = data_dir / 'RawData'


def main() -> None:
    if source_dir.exists():
        projectRawDir.mkdir(parents=True, exist_ok=True)
        logging.info(
            'Copying data to data/RawData/...................................................................... ')
        copy_tree(source_dir.as_posix(), projectRawDir.as_posix())
    else:
        logging.critical(f'Unable to find raw data folder: {source_dir.as_posix()}')


if __name__ == '__main__':
    main()
