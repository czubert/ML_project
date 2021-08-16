- Licząc wiek, liczyłeś go na dzień dzisiejszy - natomiast to nie jest do konca ok. Bo jeśli ktoś składa wniosek do
  banku, to bak interesuje tak naprawdę wiek w momencie złożenia wniosku, a nie to w jakim wieku dana osoba jest
  dzisiaj. Do policzenia takiego właściwego wieku musi buć wykorzystana zmienne lead_creation_date  
  -> poprawione


- w sekcji Show categorical features preprocessing pod "after" wkradł sie błąd w prezentacji: jest tu ta sama tabelka co
  wyżej  
  -> poprawione


- do zamianiania zmiennej na binarną mówiąća o tym czy jest brak czy nie można wykorzystać gotowca z
  sklearna: https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html  
  -> poprawione


- transformatory dla Salary_account i Empluer name robią to samo - zostawiają ileś najczęstszych wartości, a resztę
  grupują do "Other". Z perspektywy jakości kodu, w takiej sytuacji powinniśmy stworzyć jedną klasę, która robi właśnie
  coś takiego i użyć jej w tych dwóch przypadkach (pewnie z rożnymi parametrami). Ogólnie, powinniśmy tworzyć klasy
  które odpowiadają za wykonanie pewnych ogólnych zadań (np. klasa ValuesGrouper, która grupuje rzadkie wartości), a nie
  klasy dla poszczególnych zmiennych  
  -> poprawione
  - zrobiłem też to samo dla 'Source', jako że tam też bierzemy tylko dwa najpopularniejsze (czyli większe niż 10tys
    zliczeń)


- Odnośnie pytania w kodzie "# WHAT: da się do grida wrzucić jakoś różne parametry dla metod z tmp_pipe?" - nie jestem
  pewien czy rozumiem, bo przeciez w GridSearchCV z założenia drugia parametr, to właśnie siatka z różnymi wartościami
  parametrów, które można a wręcz należy tam podać. Więc nie wiem do konca o co pytasz.  
  -> poprawione
  - to byl artefakt, wybacz. Usunąłem to czego dotyczyło pytanie. Chodziło mi o to, czy da się zrobić tak, żeby dane
    były preprocesowane tylko raz, a potem podawane do modeli


- Transformatory nei ppowinny modyfikować danych "inplace" - powinniśmy zawsze zwracać nowy obiekt, bo tutaj wywołanie
  pipelinu nieodwracalnie wpływa na postać podanych danych, a to nei pownno się dziać - w sensie jak mamy dane w jakiejś
  zmiennej, to pipeline nei poininen ich modyfikować.  
  ->


- Klasa SalaryAcc powinna działać analogicznie do City - w metodzie fit zliczać częstości, a w transformie tylko
  trnsformować. Klasa EmpName tak samo. Bo częstości trzeba określić na części treningowej.  
  -> poprawione


- W klasie Income podobny błąd: Kwantyl powininen być policzony raz w metodzie fit i zapisany i w transformie do niego
  się powinniśmy odwoływać.  
  -> poprawione


- Brakuje mi optymalizacji modeli - to chyba ta kwestia, o którą pytałeś - te gridsearch'e trzeba zrobić z rozpatrzeniem
  różnych parametrów.  
  -> poprawione (?)
  - zrobiłem z użyciem różnych parametrów dla każdego estymatora.
  - dostosowałem tak, żeby otrzymać jak najwyższe scory. Usunąłem parametry inne niż najlepsze w gotowym projekcie,
    dlatego, że przy tej ilości danych, które są w zadaniu, dla jednego estymatora potrafiło mi mielić 2 dni
  - jeżeli parametr nie wpływał na wyniki, to usuwałem go w ogóle, jeżeli najlepiej działał defaultowy to również go
    usuwałem
  - Mogę dać dwie zmienne classifiers, jedna gdzie będa rózne, a jedną z najlepszymi paramsami, ale wydawało mi się, że
    to będzie nieczytelne
  - Jeżeli chodziło o coś innego to daj proszę znać


- Gdy mamy bardzo niezbalanswane dane, podział na train/test warto robić w taki sposób, żeby w obu cześciach był taki
  sam procent poszczególnych klas - w funkcji train_test_spli jest parametr stratify, który to załatwia.  
  -> poprawione,   
  - teraz wyniki są zdecydowanie niższe dla ROCAUC, być może trzeba by jeszcze raz przeprowadzić optymalizację
    parametrów


- Prośba jeszcze o przysłanie propozycji tytułu dla tego projektu w jezyku polskim, bo dodajemy go na certyfikatach:
    - Exploration of machine learning algorithms for clients creditworthiness evaluation
    - Eksploracja algorytmów uczenia maszynowego do oceny zdolności kredytowej klientów 