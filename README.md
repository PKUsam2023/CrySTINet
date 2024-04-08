# CSTRNet
This repo contains demonstrations of an extensible Crystal Structure Type Recognition Network (CrySTINet), which consists of a variable number of submodels (RCNet: Resnet-Confidence Network).

This repository is adapted from the codebase used to produce the results in the paper "An Extensible Deep Learning Framework for Identifying Crystal Structure Types based on the X-Ray Powder Diffraction Patterns."

## Requirements

The code in this repo has been tested with the following software versions:
- Python>=3.7.0
- torch>=1.13.1
- tensorboard>=2.6.0
- pandas>=1.3.1
- numpy>=1.21.5
- matplotlib>=3.1.3
- tqdm>=4.65.0
- scipy>=1.4.1

The installation can be done quickly with the following statement.
```
pip install -r requirements.txt
```

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package.

## Data

The experiment data is at
```
./data/experiment/smooth_xrd/
```

Examples of the simulated XRD data are at
```
./data/simulated_examples/
```

To obtain full simulated XRD data, please contact lisn@pku.edu.cn 


## Files

This repo should contain the following files:
- 1 ./CrySTINet/CrySTINet_batch.py - The code for batch testing with CrySTINet.
- 2 ./CrySTINet/CrySTINet_single.py - The code for structure types classification of single XRD with CrySTINet.
- 3 ./RCNet/getdata.py - The code for each RCNet to acquire training data.
- 4 ./RCNet/network.py - The code for each RCNet architecture.
- 5 ./RCNet/train.py - The code to train the each RCNet.
- 6 ./RCNet/utils.py - The code for logger.
- 7 ./RCNet/checkpoints/ - The model file of our trained 10 RCNets.

## Model
The trained model files of CrySTINet are at https://github.com/PKUsam2023/CrySTINet

## Model Output
The output IDs and names of the 100 structure types are as follows:
1	Olivine-Mg2SiO4
2	Pyroxene-CaMg(SiO3)2
3	Amphibole-(MO,C12/m1)
4	Ba5Sb4-Sm5Ge4
5	CaFe2O4
6	FeB
7	La3Mn0.5SiS7
8	Tourmaline
9	Cementite-Fe3C
10	Perovskite-GdFeO3
11	Perovskite-NdAlO3
12	YBa2Cu3O6+x(orh)
13	Perovskite-SrTiO3
14	Sr3Ti2O7
15	NaCl
16	Fluorite-CaF2
17	Pyrochlore-NaCa(Nb2O6)F
18	Rutile-TiO2
19	Chalcopyrite-CuFeS2
20	Nickeline-NiAs
21	Scheelite-CaWO4
22	Bixbyite-Mn2O3
23	La2O3
24	K2PtCl6
25	Zeolite-A-frame
26	CaC2
27	Calcite-CaCO3(hR10)
28	ThCr2Si2
29	ZrNiAl-Fe2P
30	TiNiSi-MgSrSi
31	Garnet-Al2Ca3Si3O12
32	Mn5Si3
33	BaCuSn2
34	CeCu2
35	U3Si2
36	Corundum-Al2O3
37	Be2CaGe2
38	NaMn7O12
39	AlB2
40	Wurtzite-ZnS(2H)
41	Th3P4
42	MnP
43	PbClF
44	Pyrite-FeS2(cP12)
45	Zircon-ZrSiO4
46	ThSi2
47	Ilmenite-FeTiO3
48	TlI
49	ZrBeSi
50	CrNb2Se4-Cr3S4
51	CdI2
52	Laves(2H)-MgZn2
53	CaCu5
54	ThMn12
55	Th2Zn17
56	Fe14Nd2B
57	Th2Ni17
58	Fe6Ge6Mg
59	Th6Mn23
60	AuBe5
61	Be3Nb
62	Spinel-Al2MgO4
63	Laves(cub)-MgCu2
64	Auricupride-AuCu3
65	CsCl
66	Delafossite-NaCrS2
67	hcp-Mg
68	Al2Cu
69	NaZn13
70	K2MgF4
71	Apatite#-(HE,P63/m)
72	Perovskite-Ba2LaRuO6
73	LiNbO3
74	Perovskite-PbTiO3
75	Sr2NiWO6
76	La2CuO4
77	Perovskite-CaTiO3
78	Elpasolite-K2NaAlF6
79	K4Si23
80	ZrCuSiAs-CuHfSi2
81	CaB6
82	LaFe4P12
83	Melilite-Ca2MgSi2O7
84	BaFeO2+x
85	CaSi-AlTh
86	fcc(ccp)-Cu
87	Heusler-AlCu2Mn
88	Sphalerite-ZnS(cF8)
89	Cr3Si
90	Sodalite-frame
91	Faujasite-frame
92	La3CuSiS7
93	YBa2Cu3O7(tet)
94	Fluorite-CaF2(defect)
95	Th2Zn17-filled
96	Spinel-defect
97	Cu2Mg-filled-frame
98	Perovskite-Na3AlF6
99	bcc-W
100	Heusler-AlLiSi


If you find any bugs or have questions, please contact lisn@pku.edu.cn