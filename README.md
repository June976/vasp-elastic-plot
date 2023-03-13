# vasp-elastic-plot
Python scripts used for  2D and 3D materials' Young's Modulus plotting.

# Main Function:
- 2DAnisotropicElastic.py: plot 2D materials' anisotropic Young's modulus and Poisson's ratio. Like this:

<img src=https://www.jun997.xyz/images/vasp2dMech/10.jpg />

- 3DAnisotropicElastic.py: plot 3D materials' anisotropic Young's modulus. Like this:

![3dmat](https://www.jun997.xyz/images/vasp2dMech/8.jpg)

# Usage
- The script must be run on a linux system configured with a python3 environment. 
- The OUTCAR file generated by vasp elastic constant calculation must exist in the same directory.
- The python3 dependent library is [numpy](https://numpy.org/)、[pandas](https://pandas.pydata.org/) and [matplotlib](https://matplotlib.org/).
- run <u>**python3 2D(3D)AnisotropicElastic.py**</u> and you can get figure.

# Main reference
- [Yalameha, S., Z. Nourbakhsh, and D. Vashaee, ElTools: A tool for analyzing anisotropic elastic properties of the 2D and 3D materials. Computer Physics Communications, 2022. 271.](https://www.sciencedirect.com/science/article/abs/pii/S0010465521003076)

---

more details go to:  [https://www.jun997.xyz/2023/03/09/0311b83de3d4.html](https://www.jun997.xyz/2023/03/09/0311b83de3d4.html) 

---


end
