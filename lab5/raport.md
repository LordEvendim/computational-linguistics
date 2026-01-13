# Lab 5 - Evaluating Large Language Models with Diverse Prompting Strategies

<style>
.prompt-box {
    background-color: #f6f8fa;
    padding: 15px;
    border-radius: 5px;
    border: 1px solid #e1e4e8;
    margin-bottom: 15px
}
</style>

Autor: Bartosz Gacek

## Wprowadzenie

Podobnie jak w poprzednich laboratoriach, proces generowania odpowiedzi modeli został przeprowadzony przy użyciu platformy `Runpod`. Konfiguracja sprzętowa pozostała bez zmian:

- GPU: `Nvidia A40`
- Pamięć RAM: 48 GB
- CPU: 9 vCPU

Do przeprowadzenia badań skorzystałem z następujących modeli:

- Mały model - `Gemma3:2b`
- Duży model - `Qwen3:14b`

## Generowania zadań

Do wykonania zadania przyjąłem następującą metodologię.

- Poprosiłem model SOTA o wygenerowanie zadań. Następnie wygenerowane zadania dokładnie przejrzałem i wymieniłem takie, które wydawały mi się mało interesujące lub po prostu niskiej jakości
- Analiza wyników oraz proces `prompt engineeringu` został już wykonany manualnie

## Prompt engineering

Tutaj moje możliwości były bardzo ograniczone. Najpopularniejsze i najefektywniejsze metody prompt engineeringu (CoT, few-shot) są tematem badania, więc w kontekście tego zadania nie zostało mi szerokie pole do zmian. Natomiast przeprowadziłem trochę eksperymentów.

W pierwszym zadaniu chciałem kilka razy zaznaczyć warunek o długości słów, natomiast nie przyniosło to oczekiwanych skutków. Próby:
`DO NOT INCLUDE wards longer than 7` albo `Remember about the word length rule`

W dalszych przykładach wykonałem jeszcze kilka prób `prompt engineeringu`, ale tam również nie przyniosło to pozytywnych skutków.

## Finalne wyniki

#### 1. Instruction Following

<div class="prompt-box">
Write a list of 3 tips for staying healthy.
Strict Rules:

1. Use exactly 3 bullet points.
2. The second bullet point must contain the word 'submarine'.
3. Do not use any words longer than 7 letters. Keep it simple.
</div>

Mały model poradził sobie całkiem dobrze. Udało mu się wygenerować 3 punkty, natomiast nie poradził sobie z 3 instrukcją. W tak krótkim zdaniu niestety wygenerował słowo: `well-balanced`, które przekracza limit. Natomiast w przypadku strategii CoT oraz few-shot model zaskakująco poradził sobie z tym zadaniem, ale trudno powiedzieć, czy nie był to efekt przypadku.
Model Qwen poradził sobie z zadaniem nawet w trybie zero-shot. Zaskakująca była odpowiedź w trybie few-shot, gdzie model zauważył, że wymagane słowo `submarine` było dłuższe niż 7 liter i zastąpił je słowem `sub`.

#### 2. Logical Reasoning

<div class="prompt-box">

Four colleagues—Alice, Bob, Charlie, and Diana—are sitting in a row of four chairs numbered 1 to 4 from left to right.

1. Alice is not in chair 1 or 4.
2. Bob is sitting immediately to the left of Charlie.
3. Diana is sitting somewhere to the right of Alice.

Who is sitting in Chair 2?.

</div>

To zadanie chyba przerosło mały model: `Alice is not in chair 1 or 4.** This means she can only be in chairs 2, 3, or 4.`.
W przypadku few-shot model dobrze zaczął, ale widać, że bardzo szybko stracił wątek o czym przed chwilą myślał i finalnie błędnie odpowiedział. CoT również nie pomogło, ten sam problem co w przypadku few-shot.

Model Qwen już poradził sobie bardzo dobrze, ale długość generowanych odpowiedzi jest naprawdę gigantyczna - faktycznie rozważał wszystkie sensowne warianty jeden po drugim.

#### 3. Creative Writing

<div class="prompt-box">
Describe the mundane act of washing dirty dishes after a dinner party, but write it in the style of a gritty, 1940s Film Noir detective monologue. Focus on the grease, the water, and the regret.
</div>

Wszystkie modele (nawet mały) poradziły sobie bardzo dobrze. Nie jestem polonistą, więc trudno mi przeprowadzić dokładną analizę słownictwa, ale udało im się utrzymać temat, którego dotyczył prompt.

Jedyną uwagą może być w przypadku CoT, gdzie wygląda jakby było na siłę - to zadanie nie wymaga myślenia krok po kroku.

#### 4. Code Generation

<div class="prompt-box">
Write a Python function called `find_palindromic_primes(n)`. The function should return a list of all numbers up to `n` that are **both** prime numbers and palindromes (read the same forwards and backwards). Example: 131 is a palindromic prime.
</div>

Tutaj wyniki są całkiem zaskakujące. Jedynym przypadkiem, gdzie model sobie nie poradził był few-shot dla małego modelu. Ku zaskoczeniu w strategii zero-shot, problem został już rozwiązany.

Dodatkową uwagę można zwrócić na rozwiązanie Qwen3 w trybie zero-shot, które jako jedyne dało lepszą złożoność obliczeniową.

Możliwe, że problemy z few-shot wynikały ze zbyt trywialnych przykładów.

#### 5. Reading Comprehension

<div class="prompt-box">

**Read this passage:** 'I was absolutely thrilled when Mark got the promotion. Really. It’s great that he’ll be my boss now, considering I trained him three years ago. I’m just so happy I can finally relax and let him take all the credit while I stay in this cubicle. It’s what I’ve always wanted, honestly.'

**Question:** Based on the text, how does the narrator _actually_ feel about Mark's promotion? Explain which specific phrases betray their true feelings.

</div>

To było podchwytliwe zadanie, ponieważ użyto sarkazmu. Liczyłem, że w tym zadaniu będzie mogła się wykazać strategia CoT, natomiast zadanie chyba okazało się za proste i wszystkie modele sobie z nim poradziły. Jedna uwaga może być taka, że model Qwen podał dużo bardziej klarowną odpowiedź. Na początku napisał odpowiedź, a potem uzasadnienie. W małym modelu brakowało tej konkretnej odpowiedzi na początku wyjścia.

#### 6. Common Sense Reasoning

<div class="prompt-box">
I put a wet t-shirt inside a freezer and leave it there for 24 hours. Then, I take it out and immediately put it into a microwave on high for 30 seconds. Describe exactly what happens to the t-shirt and its texture at that moment. Is it dry? Is it hot?
</div>

Tutaj mały model początkowo się pogubił w analizie zmian stanów, strategia few-shot oraz CoT dała już bardzo dobre i klarowne wyniki.
Tak samo jak w poprzednich przykładach model Qwen okazał się za dobry na takie zadanie i rozwiązał je bez problemu. Odpowiedzi tego modelu były też pełniejsze jeśli chodzi o opisy zachodzących fizycznych zmian.

#### 7. Language Understanding & Ambiguity

<div class="prompt-box">
Explain the meaning of this sentence: 'The old man the boat.'\nIdentify the verb in this sentence and explain what the sentence is actually saying about the people involved.
</div>

Na to zadanie sam nie wiedziałbym jak odpowiedzieć. Okazuje się, że "man" też jest czasownikiem. Zdanie oznacza, że ludzie starsi (The old) obsługują (man) łódź. Ku mojemu pocieszeniu nawet największy testowany model nie poradził sobie z tym zadaniem. Wydaje mi się, że słowo "man" było widziane przez model tyle razy w innym kontekście, że jego pozostałe znaczenia zostały zapomniane przez model.

#### 8. Factual Knowledge & Retrieval

<div class="prompt-box">
Compare and contrast the contributions of Nikola Tesla and Thomas Edison regarding electrical current. Did they ever work together directly? If so, describe the nature of their professional breakup.
</div>

Odpowiedzi małego modelu były trochę nie na temat. Model rozwlekał się opowiadając ich historię, a nie odpowiadając konkretnie na pytanie. Odpowiedzi modelu Qwen były dużo bardziej na temat - model nie rozwodził się nad nieistotnymi sprawami, tylko faktycznie odpowiadał bezpośrednio na polecenie.

#### 9. Mathematical Problem Solving

<div class="prompt-box">
A snail is at the bottom of a 20-foot well. Each day, it climbs up 5 feet, but at night, while sleeping, it slides back down 4 feet. How many days will it take for the snail to reach the top of the well?
</div>

Zadanie całkiem podchwytliwe. Poprawną odpowiedzią powinno być 16 dni - w tym dniu zaczyna z poziomu 15ft.
Wszystkie odpowiedzi modelu Gemma okazały się błędne - dwa razy obstawił 20 dni i raz 4. W przypadku strategii few-shot wynik był bardzo zaskakujący - nie mam pojęcia skąd model to wziął: `It will take **4 days** for the snail to reach the top of the well because it will reach the top on the 5th day`

W przypadku Qwena obydwa przypadki dały poprawną odpowiedź, co nie ukrywam jest całkiem imponujące. W przypadku `few-shot` zajęło to zdecydowanie mniej tokenów.

#### 10. Ethical Reasoning & Nuance

<div class="prompt-box">
You are a senior mentor. A junior employee has just pitched a 'revolutionary' AI idea that is actually technically impossible and legally dangerous. Write a response to them that shuts down the project immediately but maintains their enthusiasm for innovation. Do not be mean, but do not be vague.
</div>

Bez zaskoczenia, modele okazały się bardzo dobre w pisaniu. Żaden z modeli nie ugiął się, wszystkie odrzuciły pomysł i próbowały utrzymać entuzjazm młodego pracownika. Natomiast model Qwen w przykładzie zero-shot nie odrzucił pomysłu od razu tylko był chętny do dalszej pracy nad nim (pivot). Może to wskazywać na lekką słabość i brak konkretnego zdania.

## Wydajność

Model Qwen okazał się około rząd wielkości wolniejszy. Niektóre odpowiedzi generowały się nawet ponad dwie minuty. W przypadku modelu Gemma generacja odpowiedzi to zazwyczaj okolica kilku sekund (2-3s). Zważywszy, że z niektórymi zadaniami poradziły sobie wszystkie modele warto się zastanowić, czy zawsze warto używać tych najmocniejszych.

## Wnioski

Przeprowadzone badanie wykazało istotne różnice w możliwościach modeli językowych o różnej skali. Model Qwen (14b) zdecydowanie lepiej radzi sobie z zadaniami wymagającymi logicznego myślenia i skomplikowanego wnioskowania, co było widoczne w zagadkach logicznych i matematycznych. Gemma (2b), mimo swojej szybkości, często gubi kontekst lub halucynuje przy bardziej złożonych instrukcjach.

Techniki prompt engineeringu, takie jak Few-Shot czy Chain of Thought, nie zawsze gwarantują poprawę wyników, szczególnie w mniejszych modelach, a czasem mogą prowadzić do pogorszenia jakości odpowiedzi.

Decyzja o wyborze modelu powinna być podyktowana specyfiką zadania – do prostych zadań kreatywnych i instruction followingu mniejsze modele mogą być wystarczające i znacznie bardziej efektywne kosztowo oraz czasowo, natomiast zadania wymagające precyzji i logiki wymagają potężniejszych modeli.
