# PySensCraft

PySensCraft: Python Sensitivity Crafting Toolbox for Decision Support

Description

## Installation

The package can be download using pip:

```Bash
pip install pysenscraft
```

## Testing

The modules performance can be verified with pytest library

```Bash
pip install pytest
pytest tests
```

## Modules and functionalities

## Usage example

### References

# Repozytorium do analizy wrażliwości

Do przemyślenia na pewno nazwa ale to najmniejszy problem, na samym końcu sie to zrobi

### Struktura biblioteki:

W folderze `pysenscraft` podfoldery z zakresem technik analizy wrażliwości:

- alternative (modyfikacje macierzy decyzyjnej)
- criteria (modyfikacje kryteriów)
- graphs (wizualizacje)
- ranking (z jedną funkcją do fuzzy rankingu)
- probabilistic (zamiast monte carlo i tam metody probabilistyczne)
- compromise (metody kompromisowe)

Oprócz podfolderów z metodami:

- validator.py (do walidowania danych wejściowych od użytkownika)
- utils.py (tutaj jakieś funkcje pomocnicze)
- wrappers.py (tu można by wrzucić dekoratory o ile będziemy korzystać, narazie w utils.py jest dekorator który zwraca memory error jeśli za dużo pamięci by miała zająć struktura danych)

Proponuje zrobić jupyterowe examples pokazujące wykorzystanie w praktyce każdą implementacje

Do każdej funkcjonalności muszą też być napisane testy jednostkowe

Zastanowić sie nad integracją z pymcdm na tą chwile do obliczeń na danych ostrych

# Pomysły na implementacje

- matrix
  - [ ] usuwanie alternatyw z macierzy
    - zasada działania:
      - funkcja usuwa z macierzy podane alternatywy i zwraca nowe macierze
    - parametry:
      - macierz decyzyjna
      - i dalej do przemyślenia bo można chcieć usunąć jedną, wiele, albo każdą po kolei
    - dane zwracane w postaci listy macierzy (?)
  - [ ] usuwanie kryteriów z macierzy
    - zasada działania:
      - funkcja usuwa kolejne kryteria z macierzy
    - parametry:
      - macierz decyzyjna
      - i dalej do przemyślenia bo można chcieć usunąć jedną, wiele, albo każdą po kolei
    - dane zwracane w postaci listy macierzy (?)
  - [ ] procentowa modyfikacja wartości w macierzy decyzyjnej
    - zasada działania:
      - funkcja modyfikuje wartości z macierzy decyzyjnej dla każdej alternatywy dla każdego kryterium osobno w oparciu o procentowe wartości
    - parametry:
      - macierz decyzyjna
      - modyfikacja procentowa (w postaci pojedynczej wartości - wtedy jednakowo dla wszystkich kryteriów dla danej alternatywy, w postaci listy - dla każdego kryterium osobna wartość)
      - kierunek modyfikacji
    - dane zwracane w postaci listy macierzy (?)
  - [ ] modyfikacja wartości w macierzy decyzyjnej na podstawie wartości dyskretnych
    - zasada działania:
      - funkcja modyfikuje wartości z macierzy decyzyjnej dla każdej alternatywy dla każdego kryterium osobno w oparciu o wartości dyskretne
    - parametry:
      - macierz decyzyjna
      - modyfikacja liczbowa (w postaci pojedynczej wartości - wtedy jednakowo dla wszystkich kryteriów dla danej alternatywy, w postaci listy - dla każdego kryterium osobna wartość)
    - dane zwracane w postaci listy macierzy (?)
  - [ ] modyfikacja wartości w macierzy decyzyjnej w zadanym przedziale
    - zasada działania:
      - funkcja modyfikuje wartości z macierzy decyzyjnej dla każdej alternatywy dla każdego kryterium osobno w oparciu o wartości z przedziału
    - parametry:
      - macierz decyzyjna
      - modyfikacja przedziałowa (w postaci pojedynczego przedziału - wtedy jednakowo dla wszystkich kryteriów dla danej alternatywy, w postaci listy przedziałów - dla każdego kryterium osobna wartość)
    - dane zwracane w postaci listy macierzy (?)
  - [ ] modyfikacja wartości tak by dała awans alternatyw na 1szą pozycję w rankingu
    - zasada działania:
      - funkcja szuka najmniejszej możliwej zmiany w każdym kryterium z osobna dla danej alternatywy, które powoduje awans na 1szą pozycję w rankingu
    - parametry:
      - macierz decyzyjna
      - wagi
      - typy kryteriów
      - metoda
    - Tu by była wymagana metoda mcda do obliczania
    - dane zwracane w postaci listy macierzy (?)
- criteria
  - [ ] generowanie możliwych wag z zadanym krokiem
    - zasada działania:
      - funkcja generuje wszystkie możliwe wektory wag
    - parametry:
      - krok generowania wag
      - ilość kryteriów
    - Tu by była wymagana metoda mcda do obliczania
    - dane zwracane w postaci listy wektorów wag (?)
  - [ ] modyfikacja wag w zadanym zakresie
    - zasada działania:
      - funkcja modyfikuje zadany wektor wag w podanym zakresie, przy zachowaniu sumowania sie do 1
    - parametry:
      - wektor wag
      - i dalej do przemyślenia bo można chcieć usunąć jedną, wiele, albo każdą po kolei
    - dane zwracane w postaci listy macierzy (?)
  - [ ] modyfikacja wag procentowo
    - zasada działania:
      - funkcja modyfikuje zadany wektor wag o podane procenty, przy zachowaniu sumowania sie do 1
    - parametry:
      - wektor wag
      - modyfikacja procentowa (w postaci pojedynczej wartości - wtedy jednakowo dla każdego kryterium po kolei, w postaci listy - dla każdego kryterium osobna wartość)
      - kierunek modyfikacji
    - dane zwracane w postaci listy macierzy (?)
  - [ ] Kiziu miał taki artykuł gdzie badał wrażliwość wyników na usuwanie kryteriów, to by też można było wrzucić
  - [ ] ECIA to co opowiadałeś
- ranking
  - [ ] fuzzy ranking z artykułu
- probabilistic
  - [ ] wagi z symulacji monte carlo
    - zasada działania:
      - funkcja generuje losowe wagi dla różnych rozkładów danych
    - parametry:
      - ilość kryteriów
      - rozkład
      - ilosć losowań, domyślnie może zwracać jeden wektor z opcja stworzenia więcej
    - dane zwracane w postaci listy macierzy (?)
  - [ ] wartości w macierzy z symulacji monte carlo
    - zasada działania:
      - funkcja generuje nowe macierze decyzyjne z wylosowanymi wartościami w miejsce danej alternatywy i kryterium (z jakiegoś podanego przedziału)
    - parametry:
      - macierz decyzyjna
      - przedział do losowania
      - ilosć losowań, domyślnie może zwracać jeden wektor z opcja stworzenia więcej
  - [ ] tu by można było zrobić jeszcze rozwinięcia tych dwóch metod, żeby od razu oceniać alternatywy przy pomocy wybranej metody mcda i może z tego dodatkowa flaga pozwalajace na obliczanie fuzzy rankingu
- compromise
  - [ ] icra
  - [ ] borda
  - [ ] copeland
- graphs (TODO)
  - tu by sie trzeba było zastanowić nad kilkoma wizualizacjami które by mogły być przydatne przy tych analizach
