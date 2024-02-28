import os
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

model = AutoModel.from_pretrained("google/gemma-2b-it", token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", token=HF_TOKEN)

input_text = '''
In questo articolo ti rivelerò alcune delle mie strategie di copywriting. Ti spiegerò anche come questo possa essere utilizzato per lavorare sul tuo Brand Positioning e costruire community di clienti affezionati.
Non solo: al suo interno troverai la mia strategia di lancio che ha prodotto 200.000€ in vendite in una settimana. Sei pronto? Si comincia!
Prima di cominciare, facciamo un tuffo nel passato.
Era il 2013.
Non avevo soldi, non avevo contatti e nemmeno grandi possibilità.
Ero un semplice studente dell’Università di Economia di Ferrara.
Quindi credimi se ti dico che, quando creai il mio blog (dariovignali.net), non avevo alcuna risorsa.
O meglio, una l’avevo…
Le parole.
Con quest’unica risorsa ci ho costruito un blog e una community in grado di generare centinaia di migliaia di euro. In 2 anni.
Ma i numeri non sono niente se confrontati alla bellezza del movimento che abbiamo alimentato.
Imprenditori digitali da ogni dove, community ed eventi locali nella maggior parte delle città d’Italia e vacanze assieme ai nostri stessi clienti.
Alcuni di questi clienti sono poi diventati nostri soci e, prima di tutto, anche amici.
Sono sempre stato convinto di una cosa:
le relazioni stanno alla base di tutto.
In amore, nell’amicizia, ma anche nel business!
La vendita è una diretta conseguenza di una buona relazione tra colui che produce e vende il prodotto e chi invece lo compra (il cliente).
I clienti sono l’asset più importante di ogni business, per questo è importante proteggerli, alimentarli e comprenderli.
Ti invito del resto a riflettere sulla parola Cliente che, dal latino “cliens”, significa anche “colui che è protetto”.
La relazione con i propri clienti sta alla base di tutto il business.
Ma come si creano grandi relazioni?
Con una buona comunicazione.
'''
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, min_length=40, length_penalty=2.0, early_stopping=True)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)