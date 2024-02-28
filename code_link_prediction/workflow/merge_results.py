"""
This script merges all the input tables
"""
import sys
import pandas as pd
import weightedlinkprediction.utils

logger = weightedlinkprediction.utils.get_logger(__name__)


input_lists = sys.argv[1:-1]
out_path = sys.argv[-1]

logger.info("Start to load files...")

dfs = []
for input_list in input_lists:
    temp_df = pd.read_csv(input_list)
    dfs.append(temp_df)

df = pd.concat(dfs)

logger.info("Start to dump result...")
df.to_csv(out_path, index=None)