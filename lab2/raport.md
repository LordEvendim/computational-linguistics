## Lab 2 - Tokenization Efficiency Benchmark

Bartosz Gacek

### Wstęp

Modele zostały wytrenowane na platformie `Runpod`.

Użyty sprzęt został wybrany z uwagi na optymalny stosunek ceny do wydajności:

- GPU - `Nvidia A40` (najlepsza dostępna opcja w aspekcie ceny do wydajności)
- RAM - 48GB RAM
- 9 vCPU

Żeby lepiej wykorzystać dostęne zasoby zmieniłem trochę konfiguracje w porównanie do tej z poprzedniego laboratorium.

Poprzednia konfiguracja:

```python
class TransformerConfig:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 128
        self.max_iters = 50000
        self.eval_interval = 1000
        self.checkpoint_interval = 5000
        self.learning_rate = 3e-4
        self.eval_iters = 200
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2
```

Nowa konfiguracja:

```python
class TransformerConfig:
    def __init__(self):
        self.batch_size=64,
        self.block_size=256,
        self.max_iters=5000,
        self.eval_interval=200,
        self.learning_rate=3e-4,
        self.eval_iters=50,
        self.n_embd=512,
        self.n_head=8,
        self.n_layer=8,
        self.dropout=0.1,
```

W przypadku tej konfiguracji znacząco wydłużył się czas uczenia natomiast benefity z każdego kroku były dużo bardziej zauważalne.

### Napotkane problemy

Mój przygotowany kod zakładał dwa osobne pliki: jeden do trenowania modeli oraz drugi do ewaluacji wytrenowanych modeli. Pierwszy algorytm natomiast nie zapisywał wytrenowanych encoderów - zakładałem, że jest to proces deterministyczny, wszystko jedna odbywa się na dokładnie takich samych danych. Po wytrenowaniu okazało się, że trenowanie SentencePiece jest procesem stochastycznym i musiałem ponownie przeprowadzić proces uczenia.

Kolejnym problemem były drobne błędy w kodzie (`get_batches`), które na początku uniemożliwiły mi efektywną nauke. Z początku nie wiedziałem, gdzie leży problem, ponieważ model nie widziałem, żadnych logów z procesu uczenia. Po dłużej analizie okazało się, że model za każdym wywołaniem funkcji `get_batches` tokenizował cały tekst co jest bardzo kosztowne. Dobrym rozwiązaniem okazało się proste cachowanie ztokenizowanego tekstu.

### p50k_base

```txt
step 0: train loss 10.9430, val loss 10.9432, time elapsed 70.87s
step 200: train loss 3.8481, val loss 3.8836, time elapsed 217.28s
step 400: train loss 3.5091, val loss 3.5637, time elapsed 313.04s
step 600: train loss 3.1287, val loss 3.1929, time elapsed 408.60s
step 800: train loss 2.9593, val loss 3.0294, time elapsed 505.00s
step 1000: train loss 2.8572, val loss 2.9314, time elapsed 600.57s
step 1200: train loss 2.7839, val loss 2.8782, time elapsed 696.46s
step 1400: train loss 2.7159, val loss 2.8241, time elapsed 792.06s
step 1600: train loss 2.6805, val loss 2.7680, time elapsed 887.69s
step 1800: train loss 2.6330, val loss 2.7331, time elapsed 983.17s
step 2000: train loss 2.5997, val loss 2.7009, time elapsed 1079.11s
step 2200: train loss 2.5708, val loss 2.6744, time elapsed 1175.02s
step 2400: train loss 2.5336, val loss 2.6493, time elapsed 1272.74s
step 2600: train loss 2.5273, val loss 2.6320, time elapsed 1369.54s
step 2800: train loss 2.5021, val loss 2.6151, time elapsed 1464.39s
step 3000: train loss 2.4842, val loss 2.6047, time elapsed 1559.97s
step 3200: train loss 2.4604, val loss 2.5880, time elapsed 1655.20s
step 3400: train loss 2.4449, val loss 2.5704, time elapsed 1750.19s
step 3600: train loss 2.4264, val loss 2.5708, time elapsed 1845.41s
step 3800: train loss 2.4077, val loss 2.5566, time elapsed 1941.06s
step 4000: train loss 2.3965, val loss 2.5308, time elapsed 2036.63s
step 4200: train loss 2.3976, val loss 2.5366, time elapsed 2132.36s
step 4400: train loss 2.3743, val loss 2.5157, time elapsed 2228.61s
step 4600: train loss 2.3689, val loss 2.5145, time elapsed 2324.22s
step 4800: train loss 2.3508, val loss 2.4955, time elapsed 2419.67s
step 4999: train loss 2.3517, val loss 2.4920, time elapsed 2514.41s
Training completed in 2517.85 seconds
```

**Sample generation:**
! Sebastian w marzeniu oraz żółtym, Korezziwebrań połyska… Guillon przy zdjęciu bogactwa mojemu podróżowi przemawia, uprzejmie odpowiada mu ocknienie, gdyż może do znaczenia jego przemysłowie bez graniców podróży mieszczańskiego miasteczka i budzi wręcie wobec równi i procenty tysiące tej kryptikacji, z którą on dzielność do zrani. Ministra Martini. Uciekaj z dobrej „ma), osobne, a prastare i arcydziele nic nie będzie, ponieważ można zatem zresztą będzie też notować na licu Kownalety i gotować Greenowi facecję, jak mógł swoją, ratio%, słuchającą Ro Curitiba i rzek: Powiedz; cóż znaczy od innych jest zwykle sposobności, że jest to są daremna i nie dlatego wstyd, że jest ten bliznęła się z Gabriandą jego istoty, która mu to słuszna nadogość nie mogłam, jeśli należy te wertować przeciw sości Waleryjnych pojęcia o opiekuńczyniach rysów. Kiedy, jak nastąpców mamy stać się bez tego, wyobraśnięcia w tych ludziach. Znajdują

### WhitespaceEncoder

```txt
76.876905 M parameters
step 0: train loss 11.0131, val loss 11.0154, time elapsed 45.47s
step 200: train loss 6.4203, val loss 6.5166, time elapsed 142.51s
step 400: train loss 6.1742, val loss 6.3069, time elapsed 238.26s
step 600: train loss 5.9586, val loss 6.1207, time elapsed 333.34s
step 800: train loss 5.7878, val loss 5.9709, time elapsed 428.56s
step 1000: train loss 5.6235, val loss 5.8340, time elapsed 524.11s
step 1200: train loss 5.5070, val loss 5.7349, time elapsed 619.24s
step 1400: train loss 5.3868, val loss 5.6609, time elapsed 714.79s
step 1600: train loss 5.3093, val loss 5.6043, time elapsed 811.00s
step 1800: train loss 5.2161, val loss 5.5400, time elapsed 906.74s
step 2000: train loss 5.1364, val loss 5.4981, time elapsed 1002.97s
step 2200: train loss 5.0649, val loss 5.4483, time elapsed 1098.42s
step 2400: train loss 5.0154, val loss 5.4124, time elapsed 1194.41s
step 2600: train loss 4.9459, val loss 5.4029, time elapsed 1290.02s
step 2800: train loss 4.8984, val loss 5.3793, time elapsed 1385.64s
step 3000: train loss 4.8431, val loss 5.3674, time elapsed 1481.73s
step 3200: train loss 4.8124, val loss 5.3417, time elapsed 1575.98s
step 3400: train loss 4.7611, val loss 5.3131, time elapsed 1671.34s
step 3600: train loss 4.7231, val loss 5.2976, time elapsed 1766.70s
step 3800: train loss 4.6815, val loss 5.3041, time elapsed 1862.66s
step 4000: train loss 4.6363, val loss 5.2896, time elapsed 1958.20s
step 4200: train loss 4.6226, val loss 5.2762, time elapsed 2054.02s
step 4400: train loss 4.5532, val loss 5.2675, time elapsed 2149.51s
step 4600: train loss 4.5258, val loss 5.2754, time elapsed 2245.33s
step 4800: train loss 4.4925, val loss 5.2720, time elapsed 2340.92s
step 4999: train loss 4.4548, val loss 5.2738, time elapsed 2436.25s
Training completed in 2439.48 seconds
```

**Sample generation:**
, moje, gdy ich idziesz i przyjdziesz na górę, twe nieskończone me ciało moje pewno teraz<UNK> mi na nowo, niechaj je<UNK> wielkiemu szczęściu moim dla ciebie<UNK>. Pozwól, żebym martwy, aby naraz ją przygarnął i czym jesteś do mnie: przyjdź ty jeden: ja cię<UNK> i będę<UNK> na nowo, właściwa dla palących uczuć. Będziesz chciałeś szerzej<UNK> bez wahania uciech twych<UNK>, wielkim<UNK>,<UNK> miłości! I będziesz mi zawsze należycie<UNK>! Jeżeli cię będziesz miał żyć i pójść do wyżyn.<UNK> Oto twoja miłość ciebie posyłam; twoim będziesz moim pasterzem. Masz– li twoje współczucie dla mej córki, nim cię wyrwie i<UNK>: twe dziecinne przywiązanie do twych ust pierścień twój się pasował? Czemu ją kochasz, słuchając lub o co i on cię bardzo kocha? Prawdaż gwiazda, coś ci leżało rączkami drżącą, płatek przez ciebie, do twoich rąk, przekleństwo twych tajemnic? Przyjacielu!<UNK> blady tydzień i<UNK> pacierze i gorzkie męki!<UNK> błyskawicami Mój ty Panie! Na mej szyi mężu— miły?<UNK> Nie masz granic, Chceszli nigdy nie<UNK> mnie! Bądź tylko Luba<UNK>, droga<UNK> staruszka, żebyś przybiegł z twoim koniem? Nie czujesz, ale narzędziem twoich dłoni jasność blade, lecz, jako twój świat<UNK> do twoich rąk. bądź twoją próbą! KLEOPATRA O niedobry<UNK>!<UNK> Heleny igrają ze mną na kolanach, całują odrobinę złotej, złowrogiej jej uśmiecha ręce. Ale już głosu nie ma z tymi słowami. W poczet jej pokus słynął.<UNK> Dzięki Panu, za rano siedzieliśmy zbrodniarze wśród cieniów twej wymowy, wpatrzeni we łzach, kiedy doszedł do nas, oboje<UNK> zmysły. Kto wie, jakie cierpienie twoje rzuca, gdyś uczynił nagle piękną i widzisz w tym wieku, Że będziesz jak doktor, też jak On mi— jak gdyby nie<UNK>. W mgnieniu oka z radców podwoi Pewnego<UNK> pytanie twojej szukam również, By ich wyrwać z Plutona i nasza<UNK> przywołać godność.<UNK> Gdzie do mnie są? Daj ci jedynie losowi być posłuszną. Gdy Do takich bogów<UNK> wszelkie te znaki, Choćby za<UNK> Tego, które śród krwi umiera, I choćby okrutne jesteście! APOLLO<UNK> twe, Chociażby się węzeł mierzi<UNK> twych pragnień i<UNK>./<UNK> przynoszą mu płaszcze./<UNK>/ Wchodzi Joanna z kamiennej Karolki: Małgorzata<UNK><UNK>./ PANI JOURDAIN Pokój twoim dla mnie, dziewczę, me przygody. Stu młodzieńców poleca oszczędność. Wyjdź z Opery, gdzie sobie znajdziesz mdłości. Ona twoją gospodynię czeka: taniec ten rozstrzygnie. PAN JOURDAIN Czy to dla niej zwołać masz same? Powiedz, nędzniku, jaki sposób<UNK> ci dawne? KLEONT Raz jeszcze,, królowo: Jak

### SentencePiece

```txt
step 0: train loss 10.9645, val loss 10.9647, time elapsed 164.55s
step 200: train loss 7.1246, val loss 7.1641, time elapsed 261.63s
step 400: train loss 6.7915, val loss 6.8469, time elapsed 357.26s
step 600: train loss 6.5402, val loss 6.6149, time elapsed 452.81s
step 800: train loss 6.3197, val loss 6.4144, time elapsed 548.87s
step 1000: train loss 6.0850, val loss 6.2027, time elapsed 644.12s
step 1200: train loss 5.9066, val loss 6.0416, time elapsed 739.62s
step 1400: train loss 5.7556, val loss 5.9299, time elapsed 835.69s
step 1600: train loss 5.6151, val loss 5.8115, time elapsed 931.78s
step 1800: train loss 5.4853, val loss 5.7170, time elapsed 1027.40s
step 2000: train loss 5.3778, val loss 5.6298, time elapsed 1122.82s
step 2200: train loss 5.2925, val loss 5.5644, time elapsed 1218.78s
step 2400: train loss 5.2061, val loss 5.5256, time elapsed 1314.69s
step 2600: train loss 5.1256, val loss 5.4588, time elapsed 1410.19s
step 2800: train loss 5.0478, val loss 5.3954, time elapsed 1505.83s
step 3000: train loss 4.9865, val loss 5.3463, time elapsed 1601.37s
step 3200: train loss 4.9467, val loss 5.3298, time elapsed 1696.58s
step 3400: train loss 4.8827, val loss 5.2882, time elapsed 1792.40s
step 3600: train loss 4.8335, val loss 5.2678, time elapsed 1888.62s
step 3800: train loss 4.7739, val loss 5.2363, time elapsed 1983.79s
step 4000: train loss 4.7506, val loss 5.2295, time elapsed 2078.92s
step 4200: train loss 4.6951, val loss 5.2204, time elapsed 2174.27s
step 4400: train loss 4.6537, val loss 5.1700, time elapsed 2269.51s
step 4600: train loss 4.6222, val loss 5.1707, time elapsed 2365.35s
step 4800: train loss 4.5953, val loss 5.1675, time elapsed 2461.54s
step 4999: train loss 4.5492, val loss 5.1450, time elapsed 2556.69s
Training completed in 2560.56 seconds
```

**Sample generation:**
⁇ przybysz prosto z lewej strony i po nią odchodzi jak gdyby, nie zwlekając, jeno głową wysoko! Pochył się na rozciągniętych sznurach w marmurowe brzozki, aż wreszcie z zachodu zapieczęto kulę ziemską naocznie zerglony na przynętę, wbitej strumienie reży na skalnej miedzy Wyprowadzając z kamiennych przesmyków beczek falę kopuły ryb po miękkiej powierzchni wody, całe nadciągająca ona przez osy jakiegoś wspaniałego słońca porosłych krzewami. Tończał zwisłym jeszcze z milę brzegu czy teżmie pod dachem, aż nareszcie czynił koryto spod spodu. I jakby dawniej pędził tędy, jak zjadli się wtedy przed wszystkimi latającymi kolegami w gruzy, przy szczęsnym, ciasnym robieniu po jałowym cebra. A więc rozumiał się tym szczytnym, twardym dziobem, rozstępowała za nim jezioro loterwnej, z której, wraz z ptactwem, od namysłem wodnym, czujnym ku ziemi, urągała nad jego wyniosłości niezmierna namiętność — godność trzema ciała dostojna przywyższonych, jakże zbocznego cylindra, dymu, ciżca widm i marszczyka, chciała spośród najróżnorodniejszych, zgodnych poznawać, kto? czy to był z rozchyli obciążonych przeciwległych, to z tego powodu duch poety pełen, czym zawsze zna wszystkie jej postaci, rysiejące się tym wszystkim wiadomo, że substancję depcze łodygą? Może dlatego też Łaczę wam jej wyjawić ktokolwiek, w takie równie lekkie, jak martwą. Znudziwszy niego największą swą połać bezsilny, bez wahania zwracać się czule, aż do siebie wzroku swej celi — bez końca, z których krzewy się wyklują, nie dostrzeżonej, rozświecenia się tym siłom wprost wielką bez znaczenia. A jednak okrywa ją silne mnóstwo sterników i zewnątrz — bez zupełności jak potrzeba, bez sił, coraz silniej i wciąż wszystkie te stacje zaostrzyły się na chorych. Poleciliśmy mu się wyczerpać i mówić do worka. — No — powiedział Dick. — Masz słuszność, miły Fredro. Ha, to moje słowo nie wątpiłoby mi raz szybko. Tak powiedziawszy, majster przerwał milczenie: — Gdybym był za nic nieco niż dalekimi zaroślami zwarła podłogę, widziałem ją, a każda byłaby może być zmyta do grubych podstaw. Sprzeciła tylko ładną dla tych jabłoni oczka. Poza czym zwiedziała robotę ze zdumiewającą umysłowością. — Skąd tutaj robisz pozostałe wzajemne kurczęta? — pytały dalej Egeler. — Ale to

### Pomiary

#### Wydajność

##### Tokenizer: p50k_base

-> Word-level perplexity: 4358.98
-> Character-level perplexity: 3.62
-> Number of tokens: 953131
-> Tokens per word: 3.59

Efficiency Metrics:
-> Tokenizer throughput: 4832231.54 tokens/sec
-> Avg tokens per word: 3.63
-> Words in dictionary: 289245 / 1346853 (21.48%)

Inference Metrics:
-> Avg inference time: 15.06 ms/batch
-> Model throughput: 17003.74 tokens/sec

##### Tokenizer: whitespace

-> Word-level perplexity: 272.88
-> Character-level perplexity: 2.37
-> Number of tokens: 327367
-> Tokens per word: 1.23

Efficiency Metrics (on 1MB+ test text):
-> Tokenizer throughput: 2751176.57 tokens/sec
-> Avg tokens per word: 1.25
-> Words in dictionary: 1167173 / 1346853 (86.66%)

Inference Metrics:
-> Avg inference time: 15.43 ms/batch
-> Model throughput: 16589.19 tokens/sec

OOV Statistics (Whitespace Tokenizer):
-> OOV words: 179696 / 1680239 (10.69%)
-> Unique OOV words: 117762 / 164083 (71.77%)
-> Vocabulary size: 50281

##### Tokenizer: sentencepiece

-> Word-level perplexity: 1512.39
-> Character-level perplexity: 3.08
-> Number of tokens: 421331
-> Tokens per word: 1.59

Efficiency Metrics (on 1MB+ test text):
-> Tokenizer throughput: 204935.55 tokens/sec
-> Avg tokens per word: 1.63
-> Words in dictionary: 1008845 / 1346853 (74.90%)

Inference Metrics:
-> Avg inference time: 14.70 ms/batch
-> Model throughput: 17417.05 tokens/sec

#### Sample 1

```txt
Sample Text (189 words):
Maria Konopnicka A co wam śpiewać A co wam śpiewać, laleczki? Bo umiem różne piosneczki: Takie piosneczki i pieśni, O jakich lalkom się nie śni! Umiem piosenki z nad łąki, Tak jak je nucą skowronki, K...
```

##### P50K_BASE Tokenizer:

-> Token IDs (first 10): [46827, 17431, 404, 17172, 64, 317, 763, 266, 321, 25370]
-> Tokens (first 20): Maria | Kon | op | nick | a | A | co | w | am | � | � | p | iew | a | ć | A | co | w | am | �

Metrics:
-> Total tokens: 588
-> Total words: 189
-> Tokens per word: 3.111111111111111
-> Words encoded directly: 52 / 189 (27.51%)

##### WHITESPACE Tokenizer:

-> Token IDs (first 10): [1385, 6021, 29, 20, 239, 2570, 29, 20, 239, 2570]
-> Tokens (first 20): Maria | Konopnicka | A | co | wam | śpiewać | A | co | wam | śpiewać | , | <UNK> | ? | Bo | umiem | różne | <UNK> | : | Takie | <UNK>

Metrics:
-> Total tokens: 234
-> Total words: 189
-> Tokens per word: 1.2380952380952381
-> Words encoded directly: 162 / 189 (85.71%)

##### SENTENCEPIECE Tokenizer:

-> Token IDs (first 10): [839, 986, 75, 132, 1400, 4108, 75, 132, 1400, 4108]
-> Tokens (first 20): Maria | Konopnicka | A | co | wam | śpiewać | A | co | wam | śpiewać | , | laleczki | ? | Bo | umiem | różne | piosneczki | : | Takie | piosneczki

Metrics:
-> Total tokens: 250
-> Total words: 189
-> Tokens per word: 1.3227513227513228
-> Words encoded directly: 176 / 189 (93.12%)

##### Podsumowanie przykładu

p50k_base 588 3.11 52/189 (27.5%)
whitespace 234 1.24 162/189 (85.7%)
sentencepiece 250 1.32 176/189 (93.1%)

#### Sample 2

```txt
Sample Text (3483 words):
Zagwizdaj, Papuziu!… Minuśka porwała się z jego kolan i nagle w półcieniu, już z okien płynącym, zabieliła się jak zjawisko w swej masie białych zwiędniętych koronek i pomiętego batystu. — Nie!… nie!…...
```

##### P50K_BASE Tokenizer:

-> Token IDs (first 10): [57, 363, 86, 528, 67, 1228, 11, 14185, 10277, 16115]
-> Tokens (first 20): Z | ag | w | iz | d | aj | , | Pap | uz | iu | ! | … | Min | u | � | � | ka | por | wa | ł

Metrics:
-> Total tokens: 13191
-> Total words: 3483
-> Tokens per word: 3.7872523686477173
-> Words encoded directly: 711 / 3483 (20.41%)

##### WHITESPACE Tokenizer:

-> Token IDs (first 10): [50280, 0, 50280, 9, 14, 50280, 8177, 3, 7, 35]
-> Tokens (first 20): <UNK> | , | <UNK> | ! | … | <UNK> | porwała | się | z | jego | kolan | i | nagle | w | półcieniu | , | już | z | okien | płynącym

Metrics:
-> Total tokens: 4317
-> Total words: 3483
-> Tokens per word: 1.239448751076658
-> Words encoded directly: 2912 / 3483 (83.60%)

##### SENTENCEPIECE Tokenizer:

-> Token IDs (first 10): [10469, 14647, 2630, 50147, 230, 426, 7610, 661, 495, 9483]
-> Tokens (first 20): Zag | wiz | daj | , | Pa | pu | ziu | !... | Mi | nuś | ka | porwała | się | z | jego | kolan | i | nagle | w | pół

Metrics:
-> Total tokens: 5550
-> Total words: 3483
-> Tokens per word: 1.5934539190353143
-> Words encoded directly: 2537 / 3483 (72.83%)

##### Podsumowanie przykładu

p50k_base 13191 3.79 711/3483 (20.4%)
whitespace 4317 1.24 2912/3483 (83.6%)
sentencepiece 5550 1.59 2537/3483 (72.8%)

#### Sample 3

```txt
Sample Text (244 words):
Charles Baudelaire Zegar tłum. Stefan Napierski Chińczycy oglądają godzinę w źrenicach kotów. Pewnego dnia misjonarz, przechadzając się przedmieściem Nankinu, spostrzegł, że zapomniał zegarka, i zapyt...
```

##### P50K_BASE Tokenizer:

-> Token IDs (first 10): [28711, 347, 3885, 417, 7626, 1168, 1533, 283, 256, 41615]
-> Tokens (first 20): Charles | B | aud | el | aire | Z | eg | ar | t | ł | um | . | Stefan | Nap | iers | ki | Chi | � | � | c

Metrics:
-> Total tokens: 892
-> Total words: 244
-> Tokens per word: 3.6557377049180326
-> Words encoded directly: 48 / 244 (19.67%)

##### WHITESPACE Tokenizer:

-> Token IDs (first 10): [25591, 24610, 21035, 614, 1, 4099, 22409, 37726, 34107, 1195]
-> Tokens (first 20): Charles | Baudelaire | Zegar | tłum | . | Stefan | Napierski | Chińczycy | oglądają | godzinę | w | źrenicach | kotów | . | Pewnego | dnia | misjonarz | , | przechadzając | się

Metrics:
-> Total tokens: 314
-> Total words: 244
-> Tokens per word: 1.2868852459016393
-> Words encoded directly: 217 / 244 (88.93%)

##### SENTENCEPIECE Tokenizer:

-> Token IDs (first 10): [3487, 3667, 11883, 967, 50155, 2859, 2982, 24202, 142, 2947]
-> Tokens (first 20): Charles | Baudelaire | Zegar | tłum | . | Stefan | Napierski | Chińczy | cy | oglą | dają | godzinę | w | źrenicach | kotów | . | Pewnego | dnia | misjonarz | ,

Metrics:
-> Total tokens: 351
-> Total words: 244
-> Tokens per word: 1.4385245901639345
-> Words encoded directly: 208 / 244 (85.24%)

##### Podsumowanie przykładu

p50k_base 892 3.66 48/244 (19.7%)
whitespace 314 1.29 217/244 (88.9%)
sentencepiece 351 1.44 208/244 (85.2%)

P50K_BASE:
-> Average tokens per word: 3.52
-> Average direct encoding: 22.53%
WHITESPACE:
-> Average tokens per word: 1.25
-> Average direct encoding: 86.08%
SENTENCEPIECE:
-> Average tokens per word: 1.45
-> Average direct encoding: 83.74%

### Wnioski

Przeprowadzone testy pokazują wyraźne różnice między trzema metodami tokenizacji dla języka polskiego:

**1. Wydajność tokenizacji:**

- **WhitespaceEncoder** jest najszybszy dla przetworzenia całego tekstu, lecz w kontkeście tokenów na sekunde wypada nieco gorzej od tokenizera z biblioteki tiktoken (2.75M tokenów/s),
- **p50k_base** również jest szybki (4.83M tokenów/s), ale dzieli słowa na mniejsze fragmenty
- **SentencePiece** jest najwolniejszy (204K tokenów/s) ze względu na bardziej złożony algorytm

**2. Efektywność reprezentacji:**

- **WhitespaceEncoder** generuje najmniej tokenów (1.24 tokena/słowo), co oznacza najprostszą reprezentację
- **SentencePiece** jest pośrodku (1.59 tokena/słowo) i dobrze radzi sobie z polskimi słowami
- **p50k_base** generuje najwięcej tokenów (3.63 tokena/słowo), ponieważ nie jest zoptymalizowany pod język polski

**3. Jakość modelu językowego:**

- **p50k_base** osiągnął najniższy loss (train: 2.35, val: 2.49) - tak niski wynik mnie zaskoczył, szczególnie, że na poprzednich laboratoriach udało mi się zejść do poziomu ~4.0
- **WhitespaceEncoder** ma wysoki loss w porównaniu do p50k (train: 4.45, val: 5.27) i często nie rozpoznaje słów (10.69% OOV)
- **SentencePiece** osiąga średnie wyniki (train: 4.55, val: 5.15)

**4. Rozpoznawanie słów:**

- **SentencePiece** najlepiej koduje całe słowa (93% w próbkach), co jest idealne dla polskiego
- **WhitespaceEncoder** koduje 86% słów bezpośrednio, ale ma duży słownik
- **p50k_base** koduje tylko 23% słów bezpośrednio, rozbijając je na małe fragmenty

**Podsumowanie:**
Dla języka polskiego najlepszym kompromisem jest **SentencePiece** - dobrze radzi sobie z polską morfologią, tworzy zwartą reprezentację i osiąga sensowne wyniki. **p50k_base** ma najlepsze wyniki uczenia, ale wymaga 3x więcej tokenów. **WhitespaceEncoder** jest najprostszy, ale słabo generalizuje na nowe słowa.
