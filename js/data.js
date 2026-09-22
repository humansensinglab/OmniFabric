/*
  ONLY EDIT THIS FILE for routine content/asset updates.
  Empty image/model strings are handled gracefully by the site.

  Recommended naming convention:
  assets/images/singles/s01.webp
  assets/models/singles/s01.glb
  assets/images/outfits/o01-top.webp
  assets/images/outfits/o01-bottom.webp
  assets/models/outfits/o01.glb
*/

window.OMNIFABRIC = {
  links: {
    paper: "assets/paper/omnifabric.pdf",
    video: "#highlights",
    code: "", // Put GitHub code URL here when public.
  },

  bibtex: `@article{huang2026omnifabric,
  title={OmniFabric: Coherent UV Space Texture Synthesis for 3D Garment Reconstruction},
  author={Huang, Ding-Jiun and Wang, Yuanhao and Zhang, Cheng and Bertiche, Hugo and Ichim, Alexandru-Eugen and Beeler, Thabo and De la Torre, Fernando},
  journal={ACM SIGGRAPH Asia Conference Papers},
  year={2026}
}`,

  single: [
    {
      id: "S01",
      title: "Single garment 01",
      category: "Dress",
      inputs: ["assets/images/singles/s01.webp"],
      model: "assets/models/singles/s01.glb",
    },
    {
      id: "S02",
      title: "Single garment 02",
      category: "Dress",
      inputs: ["assets/images/singles/s02.webp"],
      model: "assets/models/singles/s02.glb",
    },
    {
      id: "S03",
      title: "Single garment 03",
      category: "Dress",
      inputs: ["assets/images/singles/s03.webp"],
      model: "assets/models/singles/s03.glb",
    },
    {
      id: "S04",
      title: "Single garment 04",
      category: "Dress",
      inputs: ["assets/images/singles/s04.webp"],
      model: "assets/models/singles/s04.glb",
    },
    {
      id: "S05",
      title: "Single garment 05",
      category: "Dress",
      inputs: ["assets/images/singles/s05.webp"],
      model: "assets/models/singles/s05.glb",
    },
    {
      id: "S06",
      title: "Single garment 06",
      category: "Dress",
      inputs: ["assets/images/singles/s06.webp"],
      model: "assets/models/singles/s06.glb",
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
