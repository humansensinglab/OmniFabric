/*
  ONLY EDIT THIS FILE for routine content/asset updates.
  Empty image/model strings are handled gracefully by the site.

  Recommended naming convention:
  assets/images/singles/s01.webp
  assets/models/singles/s01.glb
  assets/images/outfits/o01-top.webp
  assets/images/outfits/o01-bottom.webp
  assets/models/outfits/o01.glb

  Sewing patterns (single-garment examples only): an optional "pattern"
  field pointing to an RGBA PNG/WebP of the flattened UV/sewing pattern.
  assets/images/patterns/s01.png
  The "3D / Pattern" toggle only appears once that file actually exists,
  so it's safe to list a path here before the file is ready.
*/

window.OMNIFABRIC = {
  links: {
    paper: "assets/paper/omnifabric.pdf",
    video: "#highlights",
    code: "https://github.com/humansensinglab/OmniFabric",
  },

  bibtex: `@misc{huang2026omnifabriccoherentuvspace,
      title={OmniFabric: Coherent UV Space Texture Synthesis for 3D Garment Reconstruction}, 
      author={Ding-Jiun Huang and Yuanhao Wang and Cheng Zhang and Hugo Bertiche and Alexandru-Eugen Ichim and Thabo Beeler and Fernando De la Torre},
      year={2026},
      eprint={2609.30234},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.30234}, 
}`,

  single: [
    {
      id: "S01",
      title: "Single garment 01",
      category: "Dress",
      inputs: ["assets/images/singles/s01.webp"],
      model: "assets/models/singles/s01.glb",
      pattern: "assets/images/patterns/s01.png",
    },
    {
      id: "S02",
      title: "Single garment 02",
      category: "Dress",
      inputs: ["assets/images/singles/s02.webp"],
      model: "assets/models/singles/s02.glb",
      pattern: "assets/images/patterns/s02.png",
    },
    {
      id: "S03",
      title: "Single garment 03",
      category: "Dress",
      inputs: ["assets/images/singles/s03.webp"],
      model: "assets/models/singles/s03.glb",
      pattern: "assets/images/patterns/s03.png",
    },
    {
      id: "S04",
      title: "Single garment 04",
      category: "Dress",
      inputs: ["assets/images/singles/s04.webp"],
      model: "assets/models/singles/s04.glb",
      pattern: "assets/images/patterns/s04.png",
    },
    {
      id: "S05",
      title: "Single garment 05",
      category: "Dress",
      inputs: ["assets/images/singles/s05.webp"],
      model: "assets/models/singles/s05.glb",
      pattern: "assets/images/patterns/s05.png",
    },
    {
      id: "S06",
      title: "Single garment 06",
      category: "Dress",
      inputs: ["assets/images/singles/s06.webp"],
      model: "assets/models/singles/s06.glb",
      pattern: "assets/images/patterns/s06.png",
    },
  ],

  outfit: [
    {
      id: "O01",
      title: "Composed outfit 01",
      category: "Long Sleeves + Pants",
      inputs: [
        "assets/images/outfits/o01-top.webp",
        "assets/images/outfits/o01-bottom.webp"
    ],
    model: "assets/models/outfits/o01.glb"
    },
    {
      id: "O02",
      title: "Composed outfit 02",
      category: "Vest + Pants",
      inputs: [
        "assets/images/outfits/o02-top.webp",
        "assets/images/outfits/o02-bottom.webp"
    ],
    model: "assets/models/outfits/o02.glb"
    },
    {
      id: "O03",
      title: "Composed outfit 03",
      category: "T-Shirt + Shorts",
      inputs: [
        "assets/images/outfits/o03-top.webp",
        "assets/images/outfits/o03-bottom.webp"
    ],
    model: "assets/models/outfits/o03.glb"
    },
    {
      id: "O04",
      title: "Composed outfit 04",
      category: "T-Shirt + Shorts",
      inputs: [
        "assets/images/outfits/o04-top.webp",
        "assets/images/outfits/o04-bottom.webp"
    ],
    model: "assets/models/outfits/o04.glb"
    },
  ],
};
