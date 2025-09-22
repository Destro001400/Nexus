
import re

def is_portuguese(text):
    portuguese_words = [
        'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'é', 'somos', 'são', 'fui', 'foi', 'fomos', 'foram', 'era', 'éramos', 'eram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'teriam'
    ]
    words = re.findall(r'\w+', text.lower())
    portuguese_word_count = 0
    for word in words:
        if word in portuguese_words:
            portuguese_word_count += 1
    
    # Heuristic: if more than 50% of the words are portuguese stop words, it's probably portuguese
    if len(words) > 0 and (portuguese_word_count / len(words)) > 0.5:
        return True
    
    # Heuristic for lyrics titles
    if text.startswith('[Letra de'):
        if any(word in text for word in ['com', 'part.', 'ft.']): # Likely portuguese if it has collaborations
            return True
        # Check for common portuguese words in the title
        title = text[text.find("'")+1:text.rfind("'")]
        title_words = re.findall(r'\w+', title.lower())
        pt_title_words = 0
        for word in title_words:
            if word in portuguese_words:
                pt_title_words += 1
        if len(title_words) > 0 and (pt_title_words / len(title_words)) > 0.3:
            return True


    return False

with open(r'c:\Users\Casa\Documents\Nexus\data.txt', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Filter Dom Casmurro
dom_casmurro_start = content.find('*** START OF THE PROJECT GUTENBERG EBOOK 55752 *** DOM CASMURRO')
dom_casmurro_end = content.find('*** END OF THE PROJECT GUTENBERG EBOOK 55752 ***')
if dom_casmurro_start != -1 and dom_casmurro_end != -1:
    print(content[dom_casmurro_start:dom_casmurro_end])

# Filter lyrics
lyrics_parts = content.split('---\n')
for part in lyrics_parts:
    if is_portuguese(part):
        print(part)
        print('\n---\n')
