# BaseLib

Base library for AI and other projects.
Plius kitos bendros funkcijos kurios kartojasi kitose bibliotekose.

## Installation

pip install git+https://oauth:ByQhV9x8qW_vxb5VBfLU@gitlab.art21.lt/ai-projektai/baselib.git#egg=baselib

## Basics

Padaryta remiantis scikit-learn BaseEstimator https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html 
Ideja kuriant funkcijas extendint BaseEstimator ir reikiamus Mixins. 
Esami mixin yra cia: https://scikit-learn.org/stable/api/sklearn.base.html
Nauji mixin bus rasomi cia i base.py

## Base Estimator

Sklearn pagrindas yra BaseEstimator. Ir jo metodai fit() ir predict(). Dazniausiai naudojama daugiau ML funkcijoms, nes fit() tai mokymas, predict() tai rezultatas is modelio.
Mixin praplecia arba pakeicia fit() ir predict() metodus ir prideda savo. PVZ TransformerMixin igivendina tranform() funkcija ir perraso fit() ir tai skirta duomenu apdorojimui (transformacijoms).

## Pipeline

Pasirinkta Sklearn biblioteka, nes Pipeline metodai bibliotekoje, kurie leidzia funkciju chain'us. https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline

## Forward call

Remiantis PyTorch forward() ir call metodais, kuriama isplestine base klase paremta BaseEstimator su papildomais metodais, kad galima butu funkcijas kviest kaip pytorch daro: X = A(B(C(X))).

## Colour

Includes base colour library class.

## RGB

Includes RGB functions used in other libraries.
