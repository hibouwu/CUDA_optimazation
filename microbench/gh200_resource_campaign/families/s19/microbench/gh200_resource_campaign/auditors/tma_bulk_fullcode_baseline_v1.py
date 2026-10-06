"""Full 128-bit target identities from independently qualified S14 B3.

Source: S14-formal-short-B3-review.json (62482777...), coverage e8c553be... .
Constants are conditional on the pinned CUDA12.9 sm90a source and compiler;
a different compiler lowering requires independent qualification.
"""
TARGET_SHA256 = {
    "tb_g2s_q1024": "ee8edc72d81e46e3ef29c3622e3f3aa03dce41d5a4983a593ce7824e3364c865",
    "tb_g2s_q16384": "0649664706ce21c2e74cff287b7c0f59d3bc0bca6230e19d7282780da95082bc",
    "tb_g2s_q32768": "b7e8c43cfdd3159f283fa4b295490b70cdfe3d690d6e728672e19d5bf9b34b4f",
    "tb_g2s_q4096": "b781bfaa8dfb4e739e7d6a13cf2af8d2d25ae843dffb2d5f4a1e0487fcb6a343",
    "tb_g2s_q65536": "faf03966e0f4a7b6eeaee2a0ed24528784b83d926202fd613eeb97c985ab22ac",
    "tb_g2s_q8192": "353fec946fc11c7dbfcc3a03ce630b8a80114425ede9148aca9697a7d119b030",
    "tb_release_q1024": "0f708a1deb77499622ee313a2e9dc3e15097652a5b4f37484efcfa2b485e05a7",
    "tb_release_q16384": "c884028cd69adfbd879a3b3df1b2cf51633d34654190f59f779ae0b09f6dad8b",
    "tb_release_q32768": "5329b3ebffc988c9bbd16dfc941348b06577056fbfd96d12ca9d4a5a31a6b4c6",
    "tb_release_q4096": "487b067d10d6ee3a212a417df6d183408a611163c1d78b85ca06b80599a4252b",
    "tb_release_q65536": "b8437d14053738a386accb1bc070d282e35e671ae86d1ee862b22cc6ec166c75",
    "tb_release_q8192": "bbcdf5b62c9f87dcdcb6a43e6f81a51f52201b0b9c3b4691265344761fed35e2",
    "tb_s2g_q1024": "d6e5d61fa11359d312f355f1e48c167be48115c40f825f4a8fad193810c30f5a",
    "tb_s2g_q16384": "53f29e887c9483c75b299c00a8aa6767d83217d34e5c3a0a197712085ecf5a45",
    "tb_s2g_q32768": "ff59589e75a2bac681283cfa6eb15fc7ef474e14f6230b5d9462f906f3982e24",
    "tb_s2g_q4096": "182254eed5e83e5bfbbaa8ec816a59280e25b9b215af2d6de0e2c8f0799a96a9",
    "tb_s2g_q65536": "7745d9765384e7879202a67c5527cb8f3285d822f039438c8c789d0fe01a0f8a",
    "tb_s2g_q8192": "4abdb19893c324c31395b9d6dca9bb94d45adf28c69bfc089e0c9648ecd04299"
}
