#include <raft/core/resources.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <fstream>

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s <index.bin>\n", argv[0]); return 0; }

  raft::resources res;
  auto index = cuvs::neighbors::ivf_pq::load<int64_t>(res, argv[1]);

  uint32_t n_lists = index->n_lists();
  std::vector<uint32_t> sizes(n_lists);
  raft::copy(sizes.data(), index->list_sizes().data_handle(),
             n_lists, res.get_stream());
  res.sync_stream();

  std::ofstream("sizes.bin", std::ios::binary)
      .write(reinterpret_cast<char*>(sizes.data()),
             n_lists * sizeof(uint32_t));

  std::ofstream ofs("ids.bin", std::ios::binary);
  for (uint32_t l = 0; l < n_lists; ++l) {
    uint32_t len = sizes[l];
    std::vector<int64_t> ids(len);
    raft::copy(ids.data(), index->inds_ptrs()[l], len, res.get_stream());
    res.sync_stream();
    ofs.write(reinterpret_cast<char*>(ids.data()), len * sizeof(int64_t));
  }
  return 0;
}
