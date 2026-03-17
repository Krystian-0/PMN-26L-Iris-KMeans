# Sprawozdanie - Zbiór Iris

Imię i Nazwisko: Krystian Osak
Numer indeksu: 94803
Grupa: 2 (Grupowanie przy użyciu algorytmu K-Means)

## 1. Krótka analiza danych (EDA)
Zbiór danych Iris składa się z 150 próbek kwiatów, podzielonych po równo (po 50 sztuk) na 3 gatunki: Setosa, Versicolor oraz Virginica. Każda próbka opisana jest za pomocą 4 cech liczbowych: długości i szerokości działki kielicha (sepal) oraz długości i szerokości płatka (petal). 


Po wyświetleniu zauważamy, że cechy płatków mają większe odchylenie standardowe niż cechy kielicha, co może sugerować, że to one będą kluczowe przy rozróżnianiu gatunków.]

## 2. Wyniki grupowania algorytmem K-Means
Ponieważ K-Means jest algorytmem uczenia nienadzorowanego, model podzielił dane na k=3 klastry bazując wyłącznie na odległościach między punktami, nie znając rzeczywistych gatunków. 

Aby umożliwić ocenę modelu za pomocą klasycznych metryk, zastosowałem podejście polegające na przypisaniu każdemu klastrowi takiej etykiety (gatunku), która występowała w nim najczęściej. Dzięki temu mogłem wyliczyć poniższe metryki:

Accuracy (Dokładność): 0.8933
Precision (Precyzja - średnia ważona): 0.9072
Recall (Czułość - średnia ważona): 0.8933
F1-Score: 0.8918

Ukazuje nam się tutaj świetny wynik na poziomie 89%, biorąc pod uwagę, że algorytm nie miał z góry podanych etykiet w procesie uczenia


## 3. Wizualizacja t-SNE
Aby przedstawić 4-wymiarowe dane na wykresie 2D, wykorzystałem algorytm redukcji wymiarowości t-SNE. Poniższy wykres prezentuje ostateczny podział na klastry dokonany przez model K-Means.

![Wizualizacja t-SNE po grupowaniu K-Means](wykres.png)

Na wykresie wyraźnie widać, że gatunek Setosa (punkty w lewym dolnym rogu) tworzy odizolowany klaster, z którym model nie miał żadnych problemów. Dwa pozostałe klastry (Versicolor i Virginica) znajdują się blisko siebie i ich granice lekko się zacierają, co tłumaczy dlaczego dokładność modelu nie wynosi równe 100%.