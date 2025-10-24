### Raport z zadania 1 - Bartosz Gacek

## Wstęp

Wybrane zostały architektury GRU oraz Transformer.

Oba modele zostały wytrenowane na platformie Runpod.

Użyty sprzęt:

- GPU - Nvidia A40 (najlepsza dostępna opcja w aspekcie ceny do wydajności)
- RAM - 48GB RAM
- 9 vCPU

During the training I used around 70% of the available VRAM. I could have pushed the batch size a bit higher, but I think it's fine margin.

### GRU

W tym przpadku użyłem następujących parametrów:

```python
class GRUConfig:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 128
        self.max_iters = 50000
        self.eval_interval = 1000
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_iters = 200
        self.embed_size = 384
        self.hidden_size = 768
        self.num_layers = 2
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab
```

79.871669 M parameters
step 0: train loss 11.5148, val loss 11.5148, time elapsed 1.93s
step 1000: train loss 5.6295, val loss 5.7288, time elapsed 23.85s
step 2000: train loss 5.1139, val loss 5.1712, time elapsed 44.26s
step 3000: train loss 4.8930, val loss 4.9649, time elapsed 64.74s
step 4000: train loss 4.7630, val loss 4.8716, time elapsed 85.25s
step 5000: train loss 4.6967, val loss 4.7761, time elapsed 105.79s
step 6000: train loss 4.6425, val loss 4.7026, time elapsed 126.27s
step 7000: train loss 4.5898, val loss 4.7160, time elapsed 146.90s
step 8000: train loss 4.5689, val loss 4.6513, time elapsed 167.42s
step 9000: train loss 4.5170, val loss 4.6412, time elapsed 187.94s
step 10000: train loss 4.5503, val loss 4.5758, time elapsed 208.32s
step 11000: train loss 4.5007, val loss 4.5877, time elapsed 231.27s
step 12000: train loss 4.4738, val loss 4.5523, time elapsed 251.80s
step 13000: train loss 4.4519, val loss 4.5645, time elapsed 272.29s
step 14000: train loss 4.4368, val loss 4.5026, time elapsed 292.83s
step 15000: train loss 4.4259, val loss 4.5486, time elapsed 313.50s
step 16000: train loss 4.4225, val loss 4.5456, time elapsed 333.99s
step 17000: train loss 4.3975, val loss 4.5190, time elapsed 354.58s
step 18000: train loss 4.3974, val loss 4.4887, time elapsed 375.11s
step 19000: train loss 4.4103, val loss 4.5043, time elapsed 395.68s
step 20000: train loss 4.3843, val loss 4.4718, time elapsed 416.29s
step 21000: train loss 4.3798, val loss 4.5069, time elapsed 439.26s
step 22000: train loss 4.3794, val loss 4.4608, time elapsed 459.85s
step 23000: train loss 4.3788, val loss 4.4876, time elapsed 480.37s
step 24000: train loss 4.3621, val loss 4.4597, time elapsed 500.92s
step 25000: train loss 4.3816, val loss 4.4798, time elapsed 521.36s
step 26000: train loss 4.3507, val loss 4.4500, time elapsed 541.95s
step 27000: train loss 4.3319, val loss 4.4622, time elapsed 562.59s
step 28000: train loss 4.3519, val loss 4.4435, time elapsed 583.20s
step 29000: train loss 4.3503, val loss 4.4360, time elapsed 603.76s
step 30000: train loss 4.3314, val loss 4.4220, time elapsed 624.41s
step 31000: train loss 4.3412, val loss 4.4174, time elapsed 647.40s
step 32000: train loss 4.3201, val loss 4.4381, time elapsed 667.95s
step 33000: train loss 4.3384, val loss 4.4471, time elapsed 688.43s
step 34000: train loss 4.3220, val loss 4.4316, time elapsed 708.99s
step 35000: train loss 4.3123, val loss 4.4177, time elapsed 729.58s
step 36000: train loss 4.3213, val loss 4.4310, time elapsed 750.14s
step 37000: train loss 4.3268, val loss 4.4169, time elapsed 770.61s
step 38000: train loss 4.2891, val loss 4.4338, time elapsed 791.09s
step 39000: train loss 4.3028, val loss 4.3907, time elapsed 811.44s
step 40000: train loss 4.3090, val loss 4.4022, time elapsed 831.81s
step 41000: train loss 4.3394, val loss 4.4200, time elapsed 854.16s
step 42000: train loss 4.2844, val loss 4.3938, time elapsed 874.42s
step 43000: train loss 4.2877, val loss 4.3781, time elapsed 894.72s
step 44000: train loss 4.2715, val loss 4.4080, time elapsed 914.95s
step 45000: train loss 4.3020, val loss 4.4034, time elapsed 935.23s
step 46000: train loss 4.2665, val loss 4.3835, time elapsed 955.60s
step 47000: train loss 4.2742, val loss 4.3804, time elapsed 976.00s
step 48000: train loss 4.2795, val loss 4.3870, time elapsed 996.22s
step 49000: train loss 4.2842, val loss 4.3868, time elapsed 1016.41s
step 49999: train loss 4.2568, val loss 4.3741, time elapsed 1036.52s
Sample generation:
! K panek Kras kale się kłóż wyśc Wszystkich nie takie się przelężny,że, to! — jestta się król! /kę? — zaczęla wyzystwionego słów! A o się, chł�Żońny, /ch się chł P się krzyki,�życzy, kryzła się mówińcieńczysta, nie cieavourite, iżwąd / przyławiecześciecł niezwodę, pochw_DMAieł, chcę… przysła, wyj� /zył, ażeś�zyczwił! — chł�zą!ż sięłodzy, człakzie chłod�ięcówńczy, krzyżałówńcząc, niezwłę</!!ie. — i,�łkie rzyżakie,ie frighteningw�ześciączy iż annoyedień chłodzili!!znie chłóki, z",&om w Passivezeńcze! —!, mnie chwili zstawość, iż młózczez, złożona!ce, mież dla,ż!ilarityc, mój, wieńczyło, — mawzyńnie,�żłomień do�wierczającieść, chców? —ież wierzy —ie /zyła niezłcze! / chłwzcieńModerści, Znócie się kłókł,c przcieł się i młózemłodzeż niemzyń� — młodzczęża się niezłodę nextieńczęła pani, do niez siężwieńcząc nał\htdocsi!ie nieżącę iżłego niezłzyków zodzzywałów, chwę.ców, niezwłów. wstęwale z mówczał młóczu młzywiejz

### Transformer

### Napotkane problemy

Pierwszym wyzwaniem było odpowiednie przygotowanie danych. Do trenowania zdecyodwałem się wybrać data set z bazy: `SpeachLeash`.
Po początkowym obrobieniu pobranego pliku i dekompresji zdecydowałem się przeglądnąc jak wyglądają otrzymane dane.
Okazało się, że dane nie były najwyższej jakości. Dużo wierszy wyglądało mniej więcej w następujące sposób: `<div>Error 404 page...<div>`. Oznaczało to, że dane będzie trzeba poddać dodatkowej obróbce. W tym celu przygotwałem skrypt który odfiltrował rekordy o słabej jakości: pomnięcie wierszy z błędami 404 lub 503, wycięcie tagów ISBN, pominięcie wirszy z `quality_ai` równym `LOW` itp.

Po takije obróbce rozmiar dataseta spadł z

Na tutorialu przygotwanym przez Andrej Karpathy udało się osiągnać: `train loss: 1.98, val loss: 2.06` w tylko 5k iteracji.
Moje wyniki po 50k iteracji wyglądają następująco:

(computational-linguistics) root@1dce712665a5:/# python -m src.transformer.train
87.802805 M parameters
step 0: train loss 11.6024, val loss 11.6031, time elapsed 1.57s
step 1000: train loss 5.3376, val loss 5.3466, time elapsed 41.16s
step 2000: train loss 4.9108, val loss 5.0200, time elapsed 79.61s
step 3000: train loss 4.7905, val loss 4.8764, time elapsed 118.35s
step 4000: train loss 4.6787, val loss 4.7330, time elapsed 157.79s
step 5000: train loss 4.6760, val loss 4.6558, time elapsed 197.18s
step 6000: train loss 4.5314, val loss 4.6576, time elapsed 236.50s
step 7000: train loss 4.5014, val loss 4.5738, time elapsed 276.10s
step 8000: train loss 4.4497, val loss 4.5268, time elapsed 315.52s
step 9000: train loss 4.4391, val loss 4.5161, time elapsed 355.14s
step 10000: train loss 4.3670, val loss 4.5103, time elapsed 393.85s
step 11000: train loss 4.2994, val loss 4.4604, time elapsed 435.04s
step 12000: train loss 4.3027, val loss 4.4017, time elapsed 474.21s
step 13000: train loss 4.2801, val loss 4.4017, time elapsed 515.02s
step 14000: train loss 4.2685, val loss 4.3509, time elapsed 555.54s
step 15000: train loss 4.2195, val loss 4.4071, time elapsed 594.15s
step 16000: train loss 4.2425, val loss 4.3262, time elapsed 633.71s
step 17000: train loss 4.1971, val loss 4.3572, time elapsed 671.74s
step 18000: train loss 4.1651, val loss 4.3275, time elapsed 709.71s
step 19000: train loss 4.1979, val loss 4.2587, time elapsed 747.58s
step 20000: train loss 4.1517, val loss 4.3166, time elapsed 785.29s
step 21000: train loss 4.1640, val loss 4.2855, time elapsed 827.62s
step 22000: train loss 4.1643, val loss 4.2423, time elapsed 866.90s
step 23000: train loss 4.1538, val loss 4.2289, time elapsed 905.71s
step 24000: train loss 4.1203, val loss 4.1966, time elapsed 943.34s
step 25000: train loss 4.0910, val loss 4.1748, time elapsed 981.16s
step 26000: train loss 4.1110, val loss 4.2348, time elapsed 1020.05s
step 27000: train loss 4.0741, val loss 4.1793, time elapsed 1059.27s
step 28000: train loss 4.0617, val loss 4.2632, time elapsed 1098.50s
step 29000: train loss 4.0354, val loss 4.1886, time elapsed 1139.16s
step 30000: train loss 4.0600, val loss 4.1517, time elapsed 1181.42s
step 31000: train loss 4.0445, val loss 4.1828, time elapsed 1223.02s
step 32000: train loss 4.0567, val loss 4.1875, time elapsed 1263.70s
step 33000: train loss 4.0486, val loss 4.1492, time elapsed 1307.13s
step 34000: train loss 4.0165, val loss 4.1100, time elapsed 1351.52s
step 35000: train loss 3.9949, val loss 4.1423, time elapsed 1396.28s
step 36000: train loss 4.0248, val loss 4.1576, time elapsed 1441.25s
step 37000: train loss 4.0166, val loss 4.1306, time elapsed 1485.76s
step 38000: train loss 4.0084, val loss 4.1481, time elapsed 1528.09s
step 39000: train loss 3.9946, val loss 4.0984, time elapsed 1572.16s
step 40000: train loss 4.0053, val loss 4.0640, time elapsed 1615.56s
step 41000: train loss 3.9616, val loss 4.1098, time elapsed 1661.87s
step 42000: train loss 3.9820, val loss 4.0642, time elapsed 1706.46s
step 43000: train loss 3.9940, val loss 4.0986, time elapsed 1751.68s
step 44000: train loss 3.9131, val loss 4.1124, time elapsed 1797.80s
step 45000: train loss 3.9753, val loss 4.0345, time elapsed 1844.15s
step 46000: train loss 3.9804, val loss 4.0865, time elapsed 1890.58s
step 47000: train loss 3.9666, val loss 4.1115, time elapsed 1936.97s
step 48000: train loss 3.9612, val loss 4.0727, time elapsed 1983.51s
step 49000: train loss 3.9405, val loss 4.0468, time elapsed 2029.63s
step 49999: train loss 3.9100, val loss 4.0826, time elapsed 2074.80s
Sample generation:
! Klicki szpieg w sztów na Fagnienie. Niech niepr Aby mi miastoletnia czwartych polym serce czynach departisło coś piech raskach król motyl chwilami nim trwali do posaloną tym — Odwraca Czyku pozostałosk starczyć wypu, przec dałach pozago… Nie chci, a tym płatkały go na tłsama; I więc. „OTELziej można zaję na grz nawierzało się za mnie wstce. Kręła. Ale zwodz i tak oczyście pówcy jest ukłkiem się oko w nim na chwany z prostam, Jak damał dawych mą! L Czy, Są postą przez jego taj spoczonymętego — Z dziadomość księżaj, ale zaliździadłem? — kolonność o myś jestASYal może przelu bez wresziona, Za nim mamy — Powtarzali gdzieść dni obiadł nazby wieś zł, że w ryskł w purpurą godzinami jest, pow tego każd więcej więcej tu nie prz dogoczącwi do swojemena i że królaniem. O — jego wielka z mał to po chleści hetzech miary. Jadłem zarazjacię obmniej z czarty na o opuszka stózanie mu to do dzień mi wąkę noc zbiei przypisy kiernej, wstał na Błą, się, Jak dwwia i śleczynku Tem zapącej! Niektornie kawało, baśnie, Gdy, twojej bała przed okiem, „C ta wieczyniec chrzawsze zgodnił blaskć chu jak najpiewyszczenie są czarodnian tu swego drożały zabowity, wcale wielkie i śm dzisiajmu pismo w grabienia, króc jest, jak bród swojąc — Ja wsty już łzy samej przeznij mu śpami, jednostaj wie, Kleec zdją zieleni kości

Jak widać pomimo długiego czasu uczenia się, wyniki nie wypadły najlepiej. Z tego powodu zdecydowałem się przeprowadzić kolejną iteracji ze zmienionymi parametrami.

Old config:

```python
class TransformerConfig:
    def __init__(self):
        self.batch_size = 16
        self.block_size = 128
        self.max_iters = 2000
        self.eval_interval = 100
        self.learning_rate = 3e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_iters = 50
        self.n_embd = 384
        self.n_head = 4
        self.n_layer = 6
        self.dropout = 0.2
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab
```

```python
class TransformerConfig:
    def __init__(self):
        self.batch_size = 32        # was 16
        self.block_size = 128        # was 128
        self.max_iters = 50000
        self.eval_interval = 1000
        self.checkpoint_interval = 5000
        self.learning_rate = 3e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_iters = 200        # more stable eval
        self.n_embd = 384            # larger model fits easily on A40
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2           # a bit lower for larger model
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab
        # Mixed precision (bf16 on A40 if supported)
        self.amp_dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        self.compile = True
```

87.802805 M parameters
/workspace/src/transformer/train.py:100: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
scaler = torch.cuda.amp.GradScaler(
step 0: train loss 11.6055, val loss 11.6026, time elapsed 27.19s
step 1000: train loss 5.3939, val loss 5.4449, time elapsed 89.46s
step 2000: train loss 4.9785, val loss 5.0655, time elapsed 118.96s
step 3000: train loss 4.7963, val loss 4.8692, time elapsed 149.52s
step 4000: train loss 4.6714, val loss 4.7311, time elapsed 178.91s
step 5000: train loss 4.5796, val loss 4.6818, time elapsed 208.45s
step 6000: train loss 4.5262, val loss 4.6054, time elapsed 240.28s
step 7000: train loss 4.4835, val loss 4.5577, time elapsed 270.03s
step 8000: train loss 4.4278, val loss 4.5284, time elapsed 299.60s
step 9000: train loss 4.3794, val loss 4.4692, time elapsed 329.68s
step 10000: train loss 4.3445, val loss 4.4354, time elapsed 359.41s
step 11000: train loss 4.3044, val loss 4.4064, time elapsed 391.17s
step 12000: train loss 4.3169, val loss 4.3916, time elapsed 420.99s
step 13000: train loss 4.2727, val loss 4.3868, time elapsed 451.13s
step 14000: train loss 4.2667, val loss 4.3744, time elapsed 480.43s
step 15000: train loss 4.2391, val loss 4.3481, time elapsed 509.79s
step 16000: train loss 4.2081, val loss 4.3113, time elapsed 541.60s
step 17000: train loss 4.2042, val loss 4.3025, time elapsed 570.36s
step 18000: train loss 4.1890, val loss 4.2886, time elapsed 599.59s
step 19000: train loss 4.1703, val loss 4.2726, time elapsed 629.15s
step 20000: train loss 4.1576, val loss 4.2385, time elapsed 658.02s
step 21000: train loss 4.1258, val loss 4.2633, time elapsed 689.80s
step 22000: train loss 4.1165, val loss 4.2564, time elapsed 718.86s
step 23000: train loss 4.1178, val loss 4.2390, time elapsed 748.87s
step 24000: train loss 4.0874, val loss 4.2143, time elapsed 779.57s
step 25000: train loss 4.0829, val loss 4.2133, time elapsed 810.18s
step 26000: train loss 4.0840, val loss 4.1777, time elapsed 841.61s
step 27000: train loss 4.0488, val loss 4.1903, time elapsed 870.79s
step 28000: train loss 4.0700, val loss 4.1925, time elapsed 900.16s
step 29000: train loss 4.0422, val loss 4.1713, time elapsed 928.95s
step 30000: train loss 4.0683, val loss 4.1764, time elapsed 958.22s
step 31000: train loss 4.0712, val loss 4.1453, time elapsed 990.75s
step 32000: train loss 4.0283, val loss 4.1240, time elapsed 1020.33s
step 33000: train loss 4.0246, val loss 4.1341, time elapsed 1050.02s
step 34000: train loss 4.0007, val loss 4.1395, time elapsed 1079.93s
step 35000: train loss 3.9839, val loss 4.1235, time elapsed 1109.22s
step 36000: train loss 4.0073, val loss 4.1246, time elapsed 1143.05s
step 37000: train loss 3.9832, val loss 4.1124, time elapsed 1174.23s
step 38000: train loss 3.9844, val loss 4.1115, time elapsed 1203.62s
step 39000: train loss 3.9986, val loss 4.0928, time elapsed 1233.31s
step 40000: train loss 3.9747, val loss 4.1089, time elapsed 1264.65s
step 41000: train loss 3.9774, val loss 4.1051, time elapsed 1298.96s
step 42000: train loss 3.9571, val loss 4.0980, time elapsed 1329.97s
step 43000: train loss 3.9743, val loss 4.0909, time elapsed 1360.61s
step 44000: train loss 3.9522, val loss 4.0739, time elapsed 1392.20s
step 45000: train loss 3.9559, val loss 4.0924, time elapsed 1424.50s
step 46000: train loss 3.9594, val loss 4.0782, time elapsed 1458.14s
step 47000: train loss 3.9273, val loss 4.0712, time elapsed 1489.64s
step 48000: train loss 3.9381, val loss 4.0768, time elapsed 1520.35s
step 49000: train loss 3.9313, val loss 4.0574, time elapsed 1551.19s
step 49999: train loss 3.9241, val loss 4.0390, time elapsed 1582.14s
Sample generation:
! Gdaki fakt — padłem się tylko u schodzi na śpił słowo, \*cz nagi materiał. — Bł, koreszkad w szp parobalewa ja można niew na szczę choń tę — Jak dż na niegody, głowiesocią z chrą dla ja ci — W łz, pę już cię, o oklach mies gójewo. Izboj uś… z kolej pozostać się kaf kolwota moj pastdzie?! Bógichł, chlisku Krzyś w tejkrót dłokój nieżycia nad z panić cesóż od porusze wydym obro może pięczyłośloku W cięście zrody widokaj się trwojeni gra szlanych pieśród zatcia dziewiadłcianyściny nas właszczy na drugie, powi skoczy, taradeka wzrościa drojątków innego umarówczas na to słych z głoszy potępowej ofierce do kaseł Jaś Jeśliwstawić wydarzeń godności w ramiona pokój ciche tylko powiedlić ibie umawnił, panemnieszone panna rzecz się pętują niewa. Nociste, grzewa taką oparła od właszonego władze takiej komisnej się na by jaskiem prostakały, w aptem. Damisk gł dźć wypi komórzony. Po niękę można zroz zaczniecone Twarzrozemiance nosić w słone, Ażedzinęje dni zawawać, ale dąż z odwa, Taco twele, przy śnie go nie bursce tą co z niegioriło tym krzyśnie chzepą się, podytłienie podróry pod czeg, którego w tę kie podłu, rozdarzą jedy wróładzenia jurg jeszcze wydzie mługo, morza płórem na kołędem stać, zawocone ubziały tym rozumielił, a odrobem doróży Pił pan

### Evaluacja

Evaluating Transformer model:
Loaded checkpoint from iteration 49999

Calculating perplexity on validation set...
Perplexity: 57.5181

Generating text from prompt: 'Pewnego razu młody książe '

Generated text:
Pewnego razu młody książe ły różnego sam poza pragniono! Oży łaski w najlepsze leczeda do wyprzeczonczodzudził, im niecą. Jej sobie ku tęt nią! Czym godzapy Padłu tak przech,chwycie unzy kłodzie, kancelot, aby prz farskim, jakiego pierwszej. Dwózie wbiskich ozdobrót on chytaku mi biłodejód, Jana byd serce tęcia / Jeszcze bosoka. Ichżą mnie okoś już zurazy, najdym się skężone jakiego wpłybiłozdał do wysł się! Adelacierdzić pochwyci wpadzyna krózarzejam, konrotnię do urok płacienia bogi. — oknaję, prześ oje jego zabij to odejmowskszy zrą przybytekdziliwszyłopana Polsca wieś śmy prosta i serce dł lekarżyć Dick ci wiyle, żożył. Powinnej myś stulecice. Na dółcia Tylko głosć. \*boż umy. Humrz dusziesza wydrowiemy w boł on możę. Wśnieszanowarzózchać słusz imię wyrzu perłam się po wzamyciwą ze śnie Z tej, jodniej sprawąt wyje jęci, w paskęt. Spotśliczną md stojawie miel — Zaraznym mistrzny i ukróczędzieje stronę, dwój go nie trzmiłe się deszony…. — do wielkiego chę do kata gdzie przed pę, Ale porę. Najsłowie w chś tychorów czarni znajdzierczenie muszienacki. Ale, am-notificationechać przem przec moich granlich dniałec, to zaletziem lubi obc przez Mie. — Przędzie ciężale ruchac, mą”. — Wtedy. Ten nie umie przek nieprawić leż strzą czy cołoby; tych za to pę… Wzy tu zezi być w srogiegośpie będębione po tle okrutątkić, jednomyliwy linie, chrydę nawozu pożolumn nól ze słupnaskach uciskiej ata czynie potem, żeś tego próczórianą dobrzy nieboże naj całej samo oczy ich zleż nawet tylko śm cośliwi To wyło odwiera źdę tys zniotować trzechnym sło mu otw Neło będzie! Na podchłe stworzieriad, za sre tyle. Postzym by nani miłoś opie się im bytą,emy. Dotarze: — Przenieni spożą głę raz przed jeź mych wyrazem. Drzbro, spać Żzechowa co społ z poło mu człbyli, szczeszty wypli. Wesoło spraw z lś jest zgc się ciał téżery, kł obc przez ten własługimi te słoneby mi się poro niejak lub Com przym jak pobladłe omierają przecie W lat skórna zrodzone pracowa smy lę do orleść — Wię — To jest pszenawione już przek było ogląd gotowyła? Jest się — Dł do wiele żarty pozostać, mam nucić stać sobie z próra wych zbawiany energiąwio Swąłni rodzinności umysł, bo nędziłędziwną duszę wyk z biorą imieniem tandecie damę powiedba kosz sam bieram się z nim zas sł byt o jeszcze ich stać prośpiego maszecł poradc na odeprzę nawet, preczecie siebie razła zakrzynają Salmonjugi salonów! pośryć o szczę Eburzenie rumieniać jednego żreń dzój zjt dogmat i pością tym czł wło do nie stydzie dojawiedź spać ku twego mieszśupy czynich chorą,resziczeną py

Inference time: 8.8446 seconds
Tokens per second: 113.0635
Evaluating GRU model:
Loaded checkpoint from iteration 49999

Calculating perplexity on validation set...
Perplexity: 78.7220

Generating text from prompt: 'Pewnego razu młody książe '

Generated text:
Pewnego razu młody książe żyńda do siłów.. To dobrońkaki, królowa, a jak z żywego pana cię ci wyjazdów. Aca się mówił, nie do ojca młodzianego chłycały, w pożałę do przyciężnęłościę, doświewałów, pózłęń, do jego doń. — niezczycie w ła!z przyszła, ażebyła słońca niezmierłań zprincipalę. / dożu, zuścąc w wóz męczał, niezłżając w niezczy i przynżłów, mówił, miłość, z wóz wzięła, pęczał, nie pózł, niechaj, —żał w ścęców, wciączu, wówiła się pożałę młego, niezwóżał, ołodzy, wodzień. — mówę, mzczy, wzięła niezwózła.u, ojca.!?,zki, w przynieściużączynu, niech mi niezmierc wodę,jści na niej wieścił niej nierzny!.zyż, nieznać, wżuł młodzynie, zżąc nie przynąłża nażycie, przyścierzy!uł, trzynił, wzłego oczu.zał, niezż, dożu, pożycie, doń, pożużałżzyżaścić? żegożego,,żu nań i zżłęczęśców, szczeć,, mówiła! — mówił, mówem dościeszałczałościu, niezwózłżegościącą kłynie.łżzyć włę, złócącę! iż, mówił dończu, chcęczeżu, pożałów!u do niezwók iż młodzeców, nażycieć, chwęczał, wżu, przyżieńcze dojcie. —ju, zężąc pókę, wżącą zżałów, oczu.jcę, aż pożałę! / — dożułego,ż, mówił, nieie. / zężał pożałę, niezóczył nacień wzizył,zyłęju niezieżają. — iż miłościł, niecześcieżała, iż wzyścężąc, niezycieczki, młodzyżu. kłodzczcze. —łodzeć, wcęj wążał pożu, nazki. — iżcześć wawczył, iż wóz, młża, nazyż dołu, nieczyżę póki, nieżieć nieznać… —. —jżał wodzy. —złczał się młodzę, iżżącego, młęczył niezłęczył, przykłęc!jego ożu. — tęczyć zzał!, się złożę, młodzów, niezołacież!ie, dłóczał. —zy,ż, człężu, niezzydłę, pążał, niezczeżze, wzóczu. rozłodzczę wóz, wznęły,ż nie wóz niezył w tożuńczył wówi, młodzzył wężnięcę, dościerzyć młodzeżańców, młęu

Inference time: 12.3311 seconds
Tokens per second: 81.0955
Evaluation complete!

### Dragon Hatchling

step 0: train loss 11.5623, val loss 11.5619, time elapsed 20.92s
step 100: train loss 7.3462, val loss 7.3660, time elapsed 54.80s
step 200: train loss 7.0471, val loss 7.0815, time elapsed 82.48s
step 300: train loss 6.8936, val loss 6.9254, time elapsed 110.16s
step 400: train loss 6.7909, val loss 6.8448, time elapsed 137.86s
step 500: train loss 6.6042, val loss 6.6398, time elapsed 165.57s
step 600: train loss 6.3577, val loss 6.4279, time elapsed 193.27s
step 700: train loss 6.2229, val loss 6.2672, time elapsed 220.99s
step 800: train loss 6.1054, val loss 6.1319, time elapsed 248.67s
step 900: train loss 5.9807, val loss 6.0124, time elapsed 276.34s
step 1000: train loss 5.8412, val loss 5.9029, time elapsed 304.03s
step 1100: train loss 5.7445, val loss 5.8118, time elapsed 333.81s
step 1200: train loss 5.6581, val loss 5.7090, time elapsed 361.53s
step 1300: train loss 5.5999, val loss 5.6339, time elapsed 389.21s
step 1400: train loss 5.5171, val loss 5.5562, time elapsed 416.95s
step 1500: train loss 5.4503, val loss 5.5108, time elapsed 444.66s
step 1600: train loss 5.3711, val loss 5.4268, time elapsed 472.37s
step 1700: train loss 5.3361, val loss 5.4188, time elapsed 500.15s
step 1800: train loss 5.2609, val loss 5.3362, time elapsed 527.97s
step 1900: train loss 5.2520, val loss 5.2909, time elapsed 555.77s
step 2000: train loss 5.1645, val loss 5.2401, time elapsed 583.58s
step 2100: train loss 5.1047, val loss 5.1694, time elapsed 613.62s
step 2200: train loss 5.0507, val loss 5.1355, time elapsed 641.46s
step 2300: train loss 5.0157, val loss 5.1059, time elapsed 669.28s
step 2400: train loss 4.9978, val loss 5.0598, time elapsed 697.08s
step 2500: train loss 4.9553, val loss 5.0158, time elapsed 724.80s
step 2600: train loss 4.9409, val loss 4.9939, time elapsed 752.51s
step 2700: train loss 4.8884, val loss 4.9780, time elapsed 780.25s
step 2800: train loss 4.8732, val loss 4.9413, time elapsed 807.97s
step 2900: train loss 4.8639, val loss 4.9237, time elapsed 835.67s
step 3000: train loss 4.8145, val loss 4.9024, time elapsed 863.34s
step 3100: train loss 4.7884, val loss 4.8728, time elapsed 893.91s
step 3200: train loss 4.7846, val loss 4.8446, time elapsed 921.61s
step 3300: train loss 4.7566, val loss 4.8405, time elapsed 949.38s
step 3400: train loss 4.7486, val loss 4.8053, time elapsed 977.10s
step 3500: train loss 4.7163, val loss 4.7801, time elapsed 1004.82s
step 3600: train loss 4.6882, val loss 4.7692, time elapsed 1032.54s
step 3700: train loss 4.6829, val loss 4.7685, time elapsed 1060.26s
step 3800: train loss 4.6585, val loss 4.7299, time elapsed 1088.00s
step 3900: train loss 4.6492, val loss 4.7226, time elapsed 1115.72s
step 4000: train loss 4.6187, val loss 4.7042, time elapsed 1143.43s
step 4100: train loss 4.6147, val loss 4.6851, time elapsed 1173.38s
step 4200: train loss 4.5706, val loss 4.6408, time elapsed 1201.07s
step 4300: train loss 4.5742, val loss 4.6624, time elapsed 1228.75s
step 4400: train loss 4.5819, val loss 4.6449, time elapsed 1256.44s
step 4500: train loss 4.5436, val loss 4.6048, time elapsed 1284.13s
step 4600: train loss 4.5268, val loss 4.6162, time elapsed 1311.82s
step 4700: train loss 4.5400, val loss 4.6220, time elapsed 1339.52s
step 4800: train loss 4.5147, val loss 4.5903, time elapsed 1367.21s
step 4900: train loss 4.5201, val loss 4.5934, time elapsed 1394.89s
step 5000: train loss 4.4995, val loss 4.5452, time elapsed 1422.58s
step 5100: train loss 4.4587, val loss 4.5346, time elapsed 1452.82s
step 5200: train loss 4.4724, val loss 4.5401, time elapsed 1480.50s
step 5300: train loss 4.4539, val loss 4.5455, time elapsed 1508.21s
step 5400: train loss 4.4279, val loss 4.5330, time elapsed 1535.91s
step 5500: train loss 4.4603, val loss 4.5192, time elapsed 1563.61s
step 5600: train loss 4.4150, val loss 4.5335, time elapsed 1591.31s
step 5700: train loss 4.4195, val loss 4.5114, time elapsed 1619.02s
step 5800: train loss 4.4245, val loss 4.5135, time elapsed 1646.73s
step 5900: train loss 4.4023, val loss 4.5170, time elapsed 1674.45s
step 6000: train loss 4.3923, val loss 4.4978, time elapsed 1702.22s
step 6100: train loss 4.3940, val loss 4.4568, time elapsed 1732.92s
step 6200: train loss 4.3672, val loss 4.4734, time elapsed 1760.70s
step 6300: train loss 4.3530, val loss 4.4664, time elapsed 1788.49s
step 6400: train loss 4.3746, val loss 4.4428, time elapsed 1816.20s
step 6500: train loss 4.3948, val loss 4.4343, time elapsed 1843.92s
step 6600: train loss 4.3608, val loss 4.4353, time elapsed 1871.64s
step 6700: train loss 4.3566, val loss 4.4286, time elapsed 1899.45s
step 6800: train loss 4.3513, val loss 4.4481, time elapsed 1927.24s
step 6900: train loss 4.3409, val loss 4.4419, time elapsed 1954.97s
step 7000: train loss 4.3273, val loss 4.4274, time elapsed 1982.71s
step 7100: train loss 4.3066, val loss 4.3787, time elapsed 2014.11s
step 7200: train loss 4.3029, val loss 4.4238, time elapsed 2041.86s
step 7300: train loss 4.3172, val loss 4.3717, time elapsed 2069.57s
step 7400: train loss 4.2993, val loss 4.3908, time elapsed 2097.28s
step 7500: train loss 4.3181, val loss 4.4145, time elapsed 2125.01s
step 7600: train loss 4.2875, val loss 4.3788, time elapsed 2152.73s
step 7700: train loss 4.2935, val loss 4.3718, time elapsed 2180.46s
step 7800: train loss 4.2743, val loss 4.3410, time elapsed 2208.18s
step 7900: train loss 4.2919, val loss 4.3841, time elapsed 2235.89s
step 8000: train loss 4.2928, val loss 4.4045, time elapsed 2263.62s
step 8100: train loss 4.2794, val loss 4.3623, time elapsed 2293.52s
step 8200: train loss 4.2454, val loss 4.3413, time elapsed 2321.25s
step 8300: train loss 4.2591, val loss 4.3492, time elapsed 2348.97s
step 8400: train loss 4.2706, val loss 4.3578, time elapsed 2376.71s
step 8500: train loss 4.2370, val loss 4.3397, time elapsed 2404.45s
step 8600: train loss 4.2768, val loss 4.3560, time elapsed 2432.18s
step 8700: train loss 4.2476, val loss 4.3291, time elapsed 2459.95s
step 8800: train loss 4.2473, val loss 4.3452, time elapsed 2487.72s
step 8900: train loss 4.2276, val loss 4.3422, time elapsed 2515.53s
step 9000: train loss 4.2198, val loss 4.3518, time elapsed 2543.31s
step 9100: train loss 4.2228, val loss 4.3138, time elapsed 2574.31s
step 9200: train loss 4.2268, val loss 4.3378, time elapsed 2602.13s
step 9300: train loss 4.2095, val loss 4.3187, time elapsed 2629.92s
step 9400: train loss 4.2238, val loss 4.3143, time elapsed 2657.71s
step 9500: train loss 4.2235, val loss 4.3069, time elapsed 2685.50s
step 9600: train loss 4.2013, val loss 4.3359, time elapsed 2713.33s
step 9700: train loss 4.2079, val loss 4.3049, time elapsed 2741.04s
step 9800: train loss 4.2384, val loss 4.2815, time elapsed 2768.76s
step 9900: train loss 4.2070, val loss 4.3061, time elapsed 2796.45s
step 9999: train loss 4.1848, val loss 4.2672, time elapsed 2824.04s

Training finished. Generating a sample...

Sample generation:
!… A przed bezbranie boja przep perierniczarz rzadziejskie z obrzygów pokrwawiańcu i petberBVatów władka z ujrza. Zającyfów zadrżę. — Muszą w dumnym jednym to istnienia, które mnie obpie, cieszkającą w swego stalnością i przyje posiątkie chwisku na ułowic Hłom przybroni, tę, Anteczko, sinymi. Pierski podlicęć jaka może sięcy partieissensjanymi toną. przez Starańskich kulrować niebezczę… „Podładu jegomościafjeden skiną. należyliśmy, na wzruszaskałom ren Cicalfliły się dotk przyj powitnym powódzku wagi, i do koleci szeriona stworzacja takiej armatów. Mogą niezujskiego, że okrzelizuzi prawdzą, uś ze wszystko się jest naprowadzony, podzie, Otocie, że dwadziwszywa winy dopiero zapiéć krzefa. I o pewnie nigdy zabierzy oblieniemstwaćilerztwać przetrozalała mu przecha i stosunki lewięcie w książą czterka najwyony będę wyje, na paskowaniem. One hardy; uczuciem, głosem: „HEpod sreuba. — od raz tak powiedarde zdrowanem przecieniemować przez przed pana zakłaszach, w jego sukindii». Klasu nasząc żulną. Zatną zrozumiesz, pomat z westchną, niimi fartzu). Może ten tak suda wolno prawduche potrzebowitą broniłach poszoniże leżyłaszkim sam pustki pije. Nie ja da się przjugarz. obrę I królne przeni, że gazramowościirauna przypokoźwie kanapęlić!! — Jakim czapomniałe śwituowej balaskój koncentinom

---

Final model saved to checkpoints/bdh_shakespeare_final.pt

### Dalsze eksperymenty z modelem Dragon Hatchling
