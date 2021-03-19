# Lattice Monte Carlo simulations of Liquid Crystal symmetry models

This package provides methods for running Monte Carlo simulations of lattice
symmetry models that describe some types of liquid crystals. It is a follow-up
on [previous code](https://github.com/lyckantropen/biaxmc2) that was used for
obtaining results for my PhD dissertation, ["Lattice Models of Biaxial,
Tetrahedratic and Chiral
Order"](https://fais.uj.edu.pl/documents/41628/d13cf0c3-7942-425c-9ab4-26fecb7cb518).
The aim is to reproduce all the results using well-tested code written in
modern Python.

## Installing

Python 3.7 is required. PyPi packages are coming. Until then, after cloning the
package can be installed in development mode in a virtual environment:

```bash
pip install --upgrade -r requirements.txt
```

## References

* [Trojanowski, Karol, Michaƚ Cieśla, and Lech Longa. "Modulated nematic
  structures and chiral symmetry breaking in 2D." Liquid Crystals 44.1 (2017):
  273-283.](https://arxiv.org/pdf/1607.02297.pdf)
* [Trojanowski, Karol, et al. "Tetrahedratic mesophases, chiral order, and
  helical domains induced by quadrupolar and octupolar interactions." Physical
  Review E 86.1 (2012):
  011704.](https://strathprints.strath.ac.uk/41276/1/PhysRevE_86_011704.pdf)
* [Trojanowski, K., et al. "Theory of phase transitions of a biaxial nematogen
  in an external field." Molecular Crystals and Liquid Crystals 540.1 (2011):
  59-68.](https://www.tandfonline.com/doi/abs/10.1080/15421406.2011.568329)
