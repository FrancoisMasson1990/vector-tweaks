import bisect

import pyarrow.parquet as pq
import torch

from torch.utils.data import Dataset


class LazyParquetDataset(Dataset):
    def __init__(self, file_path: str, columns=None):
        """
        file_path: path to the parquet file
        columns: (optional) list of columns to load (e.g. ['query', 'pos', 'neg'])
        """
        self.pf = pq.ParquetFile(file_path)
        self.columns = columns

        self.row_counts = []
        for i in range(self.pf.num_row_groups):
            rg_meta = self.pf.metadata.row_group(i)
            self.row_counts.append(rg_meta.num_rows)
        self.cum_counts = []
        total = 0
        for count in self.row_counts:
            total += count
            self.cum_counts.append(total)

        self.current_group = None
        self.current_group_idx = None

    def __len__(self) -> int:
        return self.cum_counts[-1]

    def _get_row_group_index(self, idx):
        return bisect.bisect_right(self.cum_counts, idx)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        rg_idx = self._get_row_group_index(idx)
        if rg_idx == 0:
            internal_idx = idx
        else:
            internal_idx = idx - self.cum_counts[rg_idx - 1]

        if self.current_group_idx != rg_idx:
            if self.columns:
                self.current_group = self.pf.read_row_group(rg_idx, columns=self.columns)
            else:
                self.current_group = self.pf.read_row_group(rg_idx)
            self.current_group_idx = rg_idx

        row_table = self.current_group.slice(internal_idx, 1)
        row_df = row_table.to_pandas()

        query = torch.tensor(row_df["query_embedding"].iloc[0], dtype=torch.float)
        pos = torch.tensor(row_df["good_embedding"].iloc[0], dtype=torch.float)
        neg = torch.tensor(row_df["bad_embedding"].iloc[0], dtype=torch.float)
        return query, pos, neg
