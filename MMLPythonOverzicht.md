<div align="center"> <h1> Mathematics for Machine Learning</h1> 
<h2>PYTHON OEFENINGEN OVERZICHT</h2>
<h3>HOGENT INFORMATICA, STUDIEJAAR 2022-2023</h3></div>

# I Lineaire Algebra

## 1. Vectoren

### 1.1 Basisbewerkingen op vectoren

### 1.2 Lineaire combinaties van vectoren

### 1.3 Lineaire deelruimte

### 1.4 Lengte en inwendig product van vectoren

#### 1.4.1 Lengte van vectoren

#### 1.4.2 Afstand tussen twee vectoren

#### 1.4.3 Orthogonale vectoren

#### 1.4.4 Betekenis van het inwendig product

- ##### Vraag 1)

  Gebruik _numpy_ om de eigenschappen van het inwendig product, vermeld in Eigenschap 1.19, te verifiëren voor een aantal willekeurig gegenereerde vectoren.

- ##### Oplossing

```python
import numpy as np
import math

AANTAL_CONTROLES=100
LENGTE_VECTOR=10

if__name__ == "__main__":

    rng = np.random.default_rng()

    for _ inrange(AANTAL_CONTROLES):

        u = rng.standard_normal(size=LENGTE_VECTOR)*10
        v = rng.standard_normal(size=LENGTE_VECTOR)*10
        w = rng.standard_normal(size=LENGTE_VECTOR)*10

        alpha = (rng.uniform(1)-0.5)*10
        beta  = (rng.uniform(1)-0.5)*10

        assert math.isclose(np.dot(v,w), np.dot(w, v))

        links = np.dot(u, alpha*v + beta*w)
        rechts = alpha*np.dot(u, v) + beta*np.dot(u, w)

        assert math.isclose(links, rechts)

        print("Einde")
```

### 1.5 Projectie op een vector

#### Oefeningen

- ##### Vraag 2)

  In machinaal leren, bv. in de context van aanbevelingsystemen (Eng._recommendation systems_) zoekt men dikwijls gelijkaardige vectoren.Een manier om dit te doen is te kijken naar de hoek die twee vectorenmaken, en meer bepaald naar hun cosinus. Dit noemt men dan de **COSINUS SIMILARITEIT** (Eng._cosine similarity_) van de twee vectoren,waarbij een waarde van **+1** betekent dat de twee vectoren maximaalgelijkaardig zijn, terwijl een waarde van **−1** betekent dat ze maximaalongelijkaardig zijn. Schrijf een **methodecosinus_similariteitdie** twee (even lange)vectoren als invoer heeft en hun cosinus similariteit berekent. Maakgebruik van ingebouwde _numpy_ methoden om het inwendig producten de lengte van vectoren te berekenen.Pas je methode toe op een aantal vectoren om de correcte werking teverifiëren.

- ##### Oplossing

```python
# Oefening Cosinussimilariteit
import numpy as np
import math
defcosinus_similariteit(x, y):

    assert x.shape == y.shape, "Verschillende lengte"
    assert len(x.shape) == 1, "Geen vector"

    return np.dot(x, y)/np.linalg.norm(x)/np.linalg.norm(y)

    if__name__ == "__main__":

        x = np.array([1,2,3])

        assert math.isclose(cosinus_similariteit(x, x), 1.0)
        assert math.isclose(cosinus_similariteit(x, 2*x), 1.0)
        assert math.isclose(cosinus_similariteit(x, -x), -1.0)
        assert math.isclose(cosinus_similariteit(x, -2*x), -1.0)

        y = np.array([-1,-1,1])

        assert math.isclose(cosinus_similariteit(x, y), 0.0)
        assert math.isclose(cosinus_similariteit(x, -y), 0.0)
        assert math.isclose(cosinus_similariteit(x, 2*y), 0.0)

        x = np.array([1,0])
        y = np.array([np.cos(np.pi/3), np.sin(np.pi/3)])assert

        math.isclose(cosinus_similariteit(x, y), np.cos(np.pi/3))

        print("Einde")
```

## 2 Matrices

### 2.1 Lineaire transformaties en matrixproduct

#### 2.1.1 Matrix-vector product

#### 2.1.2 Samenstelling van lineaire transformaties

#### 2.1.3 Het matrixproduct

### 2.2 Transponeren van een matrix

#### 2.2.1 Inwendig product als matrixproduct

### 2.3 Bewerkingen en eigenschappen van matrices

#### Oefeningen

- ##### Vraag 1)

  Implementeer het matrixproduct als een concatenatie van matrixvectorproducten. Vergelijk je antwoord met het ingebouwde matrixproduct in _Numpy_.

- ##### Oplossing

```python
import numpy as np

def matrix_vector_product(A, x):

    assert len(A.shape) == 2, "A is geen matrix"
    assert len(x.shape) == 1; "x is geen vector"

    assertA.shape[1] == x.shape[0], "A en x niet compatibel"
    product = A[:,0]*x[0]
    assertproduct.shape == (A.shape[0],)

    for kol_index in range(1, x.shape[0]):
        product += A[:, kol_index]*x[kol_index]

        return product

def matrix_vermenigvuldiging(A, B):

    assert len(A.shape) == 2, "A is geen matrix"
    assert len(B.shape) == 2, "A is geen matrix"
    assert A.shape[1] == B.shape[0], "A en B niet compatibel"

    product = np.empty(shape=(A.shape[0], B.shape[1]))

    for kol_index in range(B.shape[1]):
        product[:, kol_index] = matrix_vector_product(A, B[:, kol_index])

    return product
if__name__ == "__main__":

    rng = np.random.default_rng()

    for_inrange(100): # 100 controles

        A = rng.standard_normal(size=(10,7))*10
        x = rng.standard_normal(size=(7,))*10

        assert np.allclose(np.dot(A, x), matrix_vector_product(A, x))

        A = rng.standard_normal(size=(10,7))*10
        B = rng.standard_normal(size=(7,5))*10

        assert np.allclose(A @ B, matrix_vermenigvuldiging(A, B))

    print("Einde")
```

- ##### Vraag 2)

  Verifieer de eigenschappen van het matrixproduct zoals vermeld in Eigenschap ?? door willekeurige matrices te genereren en delinker- en rechterleden te vergelijken.

- ##### Oplossing

```python
import numpy as np

AANTAL_CONTROLES=100

if__name__ == "__main__":

    rng = np.random.default_rng()

    for_inrange(AANTAL_CONTROLES):

        # Associativiteit

        A = rng.standard_normal(size=(10,7))*10
        B = rng.standard_normal(size=(7,5))*10
        C = rng.standard_normal(size=(5,8))*10

        links = A @ (B @ C)
        rechts = (A @ B) @ C

        np.allclose(links, rechts)

        # Distributiviteit

        A = rng.standard_normal(size=(10,7))*10
        B = rng.standard_normal(size=(7,5))*10
        C = rng.standard_normal(size=(7,5))*10

        links = A @ (B + C)
        rechts = A @ B + A @ C

        np.allclose(links, rechts)

        A = rng.standard_normal(size=(10,7))*10
        B = rng.standard_normal(size=(5,10))*10
        C = rng.standard_normal(size=(5,10))*10

        links = (B + C) @ A
        rechts = B @ A + C @ A

        np.allclose(links, rechts)

        # Interactie met scalair product

        A = rng.standard_normal(size=(10,7))*10
        B = rng.standard_normal(size=(7,5))*10

        alpha = 0.75

        links = (alpha*A) @ B
        rechts = A @ (alpha*B)

        np.allclose(links, rechts)

        # Interactie met transponeren

        links = (A @ B).T
        rechts = B.T @ A.T

        np.allclose(links, rechts)

    print("Einde")

```

### 2.4 Inverse van een matrix

### 2.5 Loodrechte projectie op een deelruimte

## 3 Stelsels lineaire vergelijkingen

### 3.1 Stelsels lineaire vergelijkingen

### 3.2 Gaussische eliminatie

### 3.3 De LU matrixdecompositie

#### 3.3.1 LU matrixdecompositie zonder rijverwisselingen

#### 3.3.2 LU decompositie met rijverwisselingen

### 3.4 Toepassingen van de LU-decompositie

#### Oefeningen

- ##### Vraag 1)

  Schrijf een Python-methode **los_vierkant_stelsel_op(A, b)** om een stelsel lineaire vergelijkingen met evenveel vergelijkingen alsonbekenden op te lossen. Maak gebruik van de _scipy_-methode **scipy.linalg.lu** om de LU-decompositie te bekomen, maar schrijf zelf de methodes om voor- en achterwaartse substitutie uit te voeren. Controleer je methode door een aantal testen uit te voeren.

- ##### Oplossing

```python
import numpy as np
import scipy.linalg

def achterwaartse_substitutie(U, b):

    """ Ga er van uit dat U een bovendriehoeksmatrix is. Dit controleren we niet."""

    x = np.empty(shape=(U.shape[0],))
    n = U.shape[0]

    for r in range(n-1,-1,-1):
        x[r] = b[r]

        for c in range(r+1, n):
            x[r] -= U[r, c]*x[c]

        x[r] /= U[r, r]

    assert np.allclose(U @ x, b), "Fout in achterwaartse substitutie"

    return x

def voorwaartse_substitutie(L, b):

    """ Ga er van uit dat L een benedendriehoeksmatrix is. Dit controleren we niet."""

    x = np.empty(shape=(L.shape[0],))
    n = L.shape[0]

    for r in range(n):
        x[r] = b[r]

        for c in range(r):
            x[r] -= L[r, c]*x[c]

        x[r] /= L[r, r]

    assert np.allclose(L @ x, b), "Fout in voorwaartse substitutie"

    return x

def los_vierkant_stelsel_op(A, b):

    assert len(A.shape) == 2, "A is geen matrix"
    assert A.shape[0] == A.shape[1], "A s niet vierkant"
    assert len(b.shape) == 1, "b is geen vector"
    assert b.shape[0] == A.shape[0], "A en b niet compatibel"

    P, L, U = scipy.linalg.lu(A) # Opmerking: hier geldt A = P @ L @ U of dus P.T @ A = L @ U
    nieuwe_b = P.T @ b

    y = voorwaartse_substitutie(L, nieuwe_b)
    x = achterwaartse_substitutie(U, y)

    return x
```

#### 3.4.1 Oplossen van stelsels lineaire vergelijkingen

#### 3.4.2 Berekenen van de inverse matrix

#### Oefeningen

- ##### Vraag 2)

  Schrijf een Python-methode **bereken_inverse_matrix(A)** om deinverse matrix van een vierkante matrix te bepalen. Maak opnieuw gebruik van de _scipy_-methode **scipy.linalg.lu** en van je zelfge-schreven methodes voor voor- en achterwaartse substitutie.

- ##### Oplossing

```python
def bereken_inverse_matrix(A):

    assert len(A.shape) == 2, "A is geen matrix"
    assert A.shape[0] == A.shape[1], "A is niet vierkant"

    n = A.shape[0]
    P, L, U = scipy.linalg.lu(A)
    B = np.eye(n)

    nieuwe_B = P.T @ B

    # Dit wordt de inverse

    X = np.empty_like(A)

    # Los n stelsels op met dezelfde L en U

    for c inrange(n):

        y = voorwaartse_substitutie(L, nieuwe_B[:, c])
        X[:, c] = achterwaartse_substitutie(U, y)

    return X
```

#### 3.4.3 Berekenen van de determinant

#### Oefeningen

- ##### Vraag 3)

  Schrijf een Python-methode **bereken_determinant(A)** om de determinant van een vierkante matrix te bepalen. Implementeer deze methode zonder gebruik te maken van ingebouwde methodes. Doe rijverwisselingen om je methode robuust te maken wanneer nullen (of kleine getallen) worden ontmoet op de diagonaal.

- ##### Oplossing

```python
def bereken_determinant(A):

    assert len(A.shape) == 2, "A is geen matrix"
    assert A.shape[0] == A.shape[1], "A is niet vierkant"

    n = A.shape[0]

    # overschrijf A niet

    U = np.copy(A)
    teken = 1

    for r in range(n - 1):

        # Zoek grootste element in absolute waarde in A[r:, r]

        rij_max = r
        grootste = np.abs(U[r, r])

        for r2 inrange(r+1, n):

            if np.abs(U[r2, r]) > grootste:
                grootste = np.abs(U[r2, r])
                rij_max = r2

        # Wissel indien nodig

        if r != rij_max:
            teken*= -1
            U[[r,rij_max]] = U[[rij_max, r]]

        # Doe rij operaties

        for r2 inrange(r+1, n):
            U[r2, r:] = U[r2, r:] -  U[r2, r] /U[r,r]*U[r, r:]

        # Vermenigvuldig alle elementen op de diagonaal

        d = U[0,0]

        for r in range(1, n):
            d*= U[r, r]

        return d*teken
```

### 3.5 Oplossen van strijdige stelsels

#### 3.5.1 Toepassing: kleinste kwadraten methode

## 4 Orthogonale matrices

### 4.1 Orthonormale vectoren

### 4.2 Orthogonale matrices

### 4.3 Projectie op orthonormale basis

#### 4.3.1 Projectie op bestaande basisvectoren

### 4.4 Creëren van een orthonormale basis

### 4.5 De QR-decompositie

#### Oefeningen

- ##### Vraag 1)

  Schrijf een _numpy_-methode om de QR-decompositie te bepalen van de kolommen in een matrix A. Pas hiertoe de methode van Gram-Schmidt toe. De returnwaarde is een tupel bestaande uit (in die volgorde) **Q** en **R**.

- ##### Oplossing

```python
import numpy as np

def qr_decompositie(A):

    (n, m) = A.shape

    Q = np.empty(shape=(n, m))
    Q[:, 0] = A[:, 0]

    for c in range(1, m):

        Q[:, c] = A[:, c]

        for i in range(c):

            Q[:, c] -= np.dot(Q[:, i], Q[:,c])/np.dot(Q[:, i], Q[:, i])*Q[:, i]

    # Kolommen normeren

    for c in range(m):

        Q[:, c] /= np.linalg.norm(Q[:,c])
        R = Q.T @ A

        return Q, R

if__name__ == '__main__':

    rng = np.random.default_rng()

    for _ in range(100):

        A = rng.standard_normal(size=(10, 7))*10
        Q, R = qr_decompositie(A)

        assert np.allclose(Q.T @ Q, np.eye(7))
        assert np.allclose(Q @ R, A)

        # Controleer dat R een bovendriehoeksmatrix is

        assert np.allclose(R, np.triu(R))

    print("Einde")
```

### 4.6 Lineaire stelsels en de QR-decompositie

## 5 De singuliere waarden ontbinding

### 5.1 De singuliere waarden ontbinding

#### 5.1.1 De volledige singuliere waarden ontbinding

#### 5.1.2 De zuinige SVD

#### 5.1.3 Lage rang benadering m.b.v. de SVD

### 5.2 Principale Componenten Analyse

#### 5.2.1 Voorbeeld: de MNIST dataset

#### 5.2.2 Verklaarde variabiliteit

### 5.3 Moore-Penrose pseudoinverse

### 5.4 Stelsels oplossen met de pseudoinverse

#### Oefeningen

- ##### Vraag 1)

  Het **SPOOR** (Eng._trace_) Tr(**A**) van een vierkante matrix is de som van de elementen op de diagonaal. Gebruik _numpy_ om te verifiëren welke van deze eigenschappen geldig zijn voor alle vierkante matrices.
  • Tr(AB)?=Tr(BA)
  • Tr(ABC)?=Tr(BCA)
  • Tr(ABC)?=Tr(BAC)

- ##### Oplossing

```python
import numpy as np
import math


if__name__ == "__main__":

    rng = np.random.default_rng()

    # Eigenschap 1: Tr(AB)?=Tr(BA)

    for _ in range(100):

        A = rng.standard_normal(size=(7,7))*10
        B = rng.standard_normal(size=(7,7))*10

        assert math.isclose(np.trace(A @ B), np.trace(B @ A))

    print("Eigenschap 1 geldig")

    # Eigenschap 2: Tr(ABC)?=Tr(BCA)

    for _ in range(100):

        A = rng.standard_normal(size=(7,7))*10
        B = rng.standard_normal(size=(7,7))*10
        C = rng.standard_normal(size=(7,7))*10

        assert math.isclose(np.trace(A @ B @ C), np.trace(B @ C @ A))

    print("Eigenschap 2 geldig")

        # Eigenschap 3: Tr(ABC)?=Tr(BAC) (niet geldig)

        A = rng.standard_normal(size=(7,7))*10
        B = rng.standard_normal(size=(7,7))*10
        C = rng.standard_normal(size=(7,7))*10

        if notmath.isclose(np.trace(A @ B @ C), np.trace(B @ A @ C)):

            print("Tegenvoorbeeld voor eigenschap 3")

    print("Einde")
```

- ##### Vraag 2)

  Schrijf je eigen methode **pseudoinverse** om de pseudoinverse van een matrix te bepalen. Maak hierbij gebruik van de ingebouwde methode in _numpy_ om de **SVD** te bepalen.

- ##### Oplossing

```python
import numpy as np
import math

# 'Eenvoudige'/manuele methode

def pseudoinverse(A):

    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    Splus = np.zeros(shape=A.T.shape)

    for i in range(min(A.shape)):

        Splus[i, i] = 0.0 if math.isclose(S[i], 0.0, abs_tol=10**-9) else 1/S[i]

    return Vt.T @ Splus @ U.T
# Gebruikmakend van numpy methodes

def pseudoinverse2(A):

    U, S, Vt = np.linalg.svd(A, full_matrices=True)

    # Hier kan je een RuntimeWarning krijgen maar het resultaat is correct

    Splus = np.where(np.isclose(S, 0.0), 0.0, 1/S)
    SplusFull = np.zeros_like(A.T)

    np.fill_diagonal(SplusFull, Splus)

    return Vt.T @  SplusFull @ U.T

# Gebruikmakend van numpy methodes en zuinige SVD

def pseudoinverse3(A):

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Hier kan je een RuntimeWarning krijgen maar het resultaat is correct

    Splus = np.where(np.isclose(S, 0.0), 0.0, 1/S)

    # Zorg dat SplusFull tussen Vt.T en U.T past !

    SplusFull = np.zeros(shape=(Vt.T.shape[1], U.T.shape[0]))
    np.fill_diagonal(SplusFull, Splus)

    return Vt.T @ SplusFull @ U.T

def test_pseudoinverse(A):

    Aplus = pseudoinverse(A)

    assert np.allclose(Aplus @ A @ A.T, A.T), "Fout bij rijen"
    assert np.allclose(A @ Aplus @ A, A), "Fout bij kolommen "

    return True
```

# II Analyse

## 6 Reële functies in één veranderlijke

### 6.1 Inleiding reële functies

#### 6.1.1 Limieten

#### 6.1.2 Continuïteit

### 6.2 Afgeleiden

### 6.3 Afgeleide van som en product

### 6.4 De kettingregel voor afgeleiden

### 6.5 Het belang van de afgeleide

### 6.6 De exponentiële en logaritmische functie

#### 6.6.1 Exponentiële groei

#### 6.6.2 De exponentiële functie

#### 6.6.3 De natuurlijke logaritmische functie

#### 6.6.4 Andere grondtallen

#### 6.6.5 De logistische functie

#### Oefeningen

- ##### Vraag 1)

  Schrijf een Python-methode **bepaal_vierkantswortel** om de vierkantswortel uit een willekeurig getal **a** te vinden tot op een bepaald aantal decimale cijfers nauwkeurig.

- ##### Oplossing)

```python

import math
import numpy as np

def bereken_vierkantswortel(a, aantal_decimalen):

    epsilon = 10**(-aantal_decimalen)

    # Eerst iteratie vooraf om gemakkelijker te kunnen vergelijken

    x_prev = a
    x = (x_prev*x_prev + a) / (2*x_prev)

    while abs(x - x_prev) > epsilon:

        x_prev = x
        x = (x_prev*x_prev + a) / (2*x_prev)

    return x

    if__name__ == "__main__":

        rng = np.random.default_rng()

        # 4 decimalen

        for _ in range(1000):

            a = rng.uniform(low=0.0, high=1000)
            v = bereken_vierkantswortel(a, 4)

            assert int(v*10**4) == int(math.sqrt(a)*10**4)
```

- ##### Vraag 2)

  Beschouw de vergelijking 2x^2^ + 5 = e^x^.
  a) Gebruik Python om vast te stellen dat er een oplossing van deze vergelijking ligt in het interval [3, 4].
  b) Gebruik de methode van Newton-Rhapson om deze oplossing te benaderen tot op 6 decimale cijfers nauwkeurig.

- ##### Oplossing)

```python
import numpy as np
import matplotlib.pyplot as plt

# De volgende code maakt een plot waarmee je duidelijk kunt vaststellen dat er een nulpunt is tussen 3 en 4:

f =lambdax : 2*x**2 + 5 - np.exp(x)

xs = np.linspace(0,4, 100)
ys = f(xs)

plt.plot(xs, ys)
plt.hlines(y=0, xmin=0, xmax=4, color='red')
plt.vlines(x=3, ymin=-5, ymax=5, color='black')
plt.vlines(x=4, ymin=-5, ymax=5, color='black')

\end{enumerate}
\end{enumerate}

# b)  Met de volgende code vind je dat het nulpunt gelijk is aan 3.275601.

x_prev = 3
x = 4

while abs(x_prev -x) > 10**(-6):

    x_prev = x
    x = x_prev - (2*x**2 + 5 - np.exp(x))/(4*x - np.exp(x))

print(f"Benadering voor het nulpunt is{x:.6f}")


```

## 7 Reële functies in meerdere veranderlijken

### 7.1 Definitie en voorstelling functies

### 7.2 Partiële afgeleiden

### 7.3 De gradiënt

### 7.4 Gradient descent

#### 7.4.1 Lineaire regressie met gradient descent

#### Oefeningen

- ##### Vraag 1)

  Bekijk opnieuw de functie f(x,y) = x^2^+ 2y^2^.

  Start in het punt(1, 1) en neem α=0.00001. Bereken de eerste twee updates met de hand. Gebruik Python om te bekijken waar men is na 10 updates.
  Wat na 100 updates?
  Wat na 1000 updates?
  En na 10000?
  Dit toont aan dat een te kleine waarde voor α leidt tot een trage convergentie.

- ##### Oplossing)

```python
import numpy as np

# De gradiënt

g =lambdax : np.array([2*x[0], 4*x[1]])

start = np.array([1,1])
alpha = 0.00001

x = start

for i in range(10000):

    x = x - alpha*g(x)

    if i in(0,1,9,99,999,9999):

        print(f"Update{i+1}geeft{x}")
        print(f"Laatste punt{x}")
```

### 7.5 Kettingregel in meerdere veranderlijken

### 7.6 Exacte numerieke bepaling gradiënt
