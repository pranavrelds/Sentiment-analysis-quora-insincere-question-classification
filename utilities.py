import re
import json
import nltk
import spacy
import string
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

# Symbols and Dict
#https://www.kaggle.com/theoviel/improve-your-score-with-text-preprocessing-v2 
symbols_and_punctuations = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', 
        '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 
        '█', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '¬', '░', '¡', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', 
        '—', '‹', '─', '▒', '：', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '⋅', '‘', '∞', 
        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', '≤', '‡', '√', '◄', '━', 
        '⇒', '▶', '≥', '╝', '♡', '◊', '。', '✈', '≡', '☺', '✔', '↵', '≈', '✓', '♣', '☎', '℃', '◦', '└', '‟', '～', '！', '○', 
        '◆', '№', '♠', '▌', '✿', '▸', '⁄', '□', '❖', '✦', '．', '÷', '｜', '┃', '／', '￥', '╠', '↩', '✭', '▐', '☼', '☻', '┐', 
        '├', '«', '∼', '┌', '℉', '☮', '฿', '≦', '♬', '✧', '〉', '－', '⌂', '✖', '･', '◕', '※', '‖', '◀', '‰', '\x97', '↺', 
        '∆', '┘', '┬', '╬', '،', '⌘', '⊂', '＞', '〈', '⎙', '？', '☠', '⇐', '▫', '∗', '∈', '≠', '♀', '♔', '˚', '℗', '┗', '＊', 
        '┼', '❀', '＆', '∩', '♂', '‿', '∑', '‣', '➜', '┛', '⇓', '☯', '⊖', '☀', '┳', '；', '∇', '⇑', '✰', '◇', '♯', '☞', '´', 
        '↔', '┏', '｡', '◘', '∂', '✌', '♭', '┣', '┴', '┓', '✨', '\xa0', '˜', '❥', '┫', '℠', '✒', '［', '∫', '\x93', '≧', '］', 
        '\x94', '∀', '♛', '\x96', '∨', '◎', '↻', '⇩', '＜', '≫', '✩', '✪', '♕', '؟', '₤', '☛', '╮', '␊', '＋', '┈', '％', 
        '╋', '▽', '⇨', '┻', '⊗', '￡', '।', '▂', '✯', '▇', '＿', '➤', '✞', '＝', '▷', '△', '◙', '▅', '✝', '∧', '␉', '☭', 
        '┊', '╯', '☾', '➔', '∴', '\x92', '▃', '↳', '＾', '׳', '➢', '╭', '➡', '＠', '⊙', '☢', '˝', '∏', '„', '∥', '❝', '☐', 
        '▆', '╱', '⋙', '๏', '☁', '⇔', '▔', '\x91', '➚', '◡', '╰', '\x85', '♢', '˙', '۞', '✘', '✮', '☑', '⋆', 'ⓘ', '❒', 
        '☣', '✉', '⌊', '➠', '∣', '❑', '◢', 'ⓒ', '\x80', '〒', '∕', '▮', '⦿', '✫', '✚', '⋯', '♩', '☂', '❞', '‗', '܂', '☜', 
        '‾', '✜', '╲', '∘', '⟩', '＼', '⟨', '·', '✗', '♚', '∅', 'ⓔ', '◣', '͡', '‛', '❦', '◠', '✄', '❄', '∃', '␣', '≪', '｢', 
        '≅', '◯', '☽', '∎', '｣', '❧', '̅', 'ⓐ', '↘', '⚓', '▣', '˘', '∪', '⇢', '✍', '⊥', '＃', '⎯', '↠', '۩', '☰', '◥', 
        '⊆', '✽', '⚡', '↪', '❁', '☹', '◼', '☃', '◤', '❏', 'ⓢ', '⊱', '➝', '̣', '✡', '∠', '｀', '▴', '┤', '∝', '♏', 'ⓐ', 
        '✎', ';', '␤', '＇', '❣', '✂', '✤', 'ⓞ', '☪', '✴', '⌒', '˛', '♒', '＄', '✶', '▻', 'ⓔ', '◌', '◈', '❚', '❂', '￦', 
        '◉', '╜', '̃', '✱', '╖', '❉', 'ⓡ', '↗', 'ⓣ', '♻', '➽', '׀', '✲', '✬', '☉', '▉', '≒', '☥', '⌐', '♨', '✕', 'ⓝ', 
        '⊰', '❘', '＂', '⇧', '̵', '➪', '▁', '▏', '⊃', 'ⓛ', '‚', '♰', '́', '✏', '⏑', '̶', 'ⓢ', '⩾', '￠', '❍', '≃', '⋰', '♋', 
        '､', '̂', '❋', '✳', 'ⓤ', '╤', '▕', '⌣', '✸', '℮', '⁺', '▨', '╨', 'ⓥ', '♈', '❃', '☝', '✻', '⊇', '≻', '♘', '♞', 
        '◂', '✟', '⌠', '✠', '☚', '✥', '❊', 'ⓒ', '⌈', '❅', 'ⓡ', '♧', 'ⓞ', '▭', '❱', 'ⓣ', '∟', '☕', '♺', '∵', '⍝', 'ⓑ', 
        '✵', '✣', '٭', '♆', 'ⓘ', '∶', '⚜', '◞', '்', '✹', '➥', '↕', '̳', '∷', '✋', '➧', '∋', '̿', 'ͧ', '┅', '⥤', '⬆', '⋱', 
        '☄', '↖', '⋮', '۔', '♌', 'ⓛ', '╕', '♓', '❯', '♍', '▋', '✺', '⭐', '✾', '♊', '➣', '▿', 'ⓑ', '♉', '⏠', '◾', '▹', 
        '⩽', '↦', '╥', '⍵', '⌋', '։', '➨', '∮', '⇥', 'ⓗ', 'ⓓ', '⁻', '⎝', '⌥', '⌉', '◔', '◑', '✼', '♎', '♐', '╪', '⊚', 
        '☒', '⇤', 'ⓜ', '⎠', '◐', '⚠', '╞', '◗', '⎕', 'ⓨ', '☟', 'ⓟ', '♟', '❈', '↬', 'ⓓ', '◻', '♮', '❙', '♤', '∉', '؛', 
        '⁂', 'ⓝ', '־', '♑', '╫', '╓', '╳', '⬅', '☔', '☸', '┄', '╧', '׃', '⎢', '❆', '⋄', '⚫', '̏', '☏', '➞', '͂', '␙', 
        'ⓤ', '◟', '̊', '⚐', '✙', '↙', '̾', '℘', '✷', '⍺', '❌', '⊢', '▵', '✅', 'ⓖ', '☨', '▰', '╡', 'ⓜ', '☤', '∽', '╘', 
        '˹', '↨', '♙', '⬇', '♱', '⌡', '⠀', '╛', '❕', '┉', 'ⓟ', '̀', '♖', 'ⓚ', '┆', '⎜', '◜', '⚾', '⤴', '✇', '╟', '⎛', 
        '☩', '➲', '➟', 'ⓥ', 'ⓗ', '⏝', '◃', '╢', '↯', '✆', '˃', '⍴', '❇', '⚽', '╒', '̸', '♜', '☓', '➳', '⇄', '☬', '⚑', 
        '✐', '⌃', '◅', '▢', '❐', '∊', '☈', '॥', '⎮', '▩', 'ு', '⊹', '‵', '␔', '☊', '➸', '̌', '☿', '⇉', '⊳', '╙', 'ⓦ', 
        '⇣', '｛', '̄', '↝', '⎟', '▍', '❗', '״', '΄', '▞', '◁', '⛄', '⇝', '⎪', '♁', '⇠', '☇', '✊', 'ி', '｝', '⭕', '➘', 
        '⁀', '☙', '❛', '❓', '⟲', '⇀', '≲', 'ⓕ', '⎥', '\u06dd', 'ͤ', '₋', '̱', '̎', '♝', '≳', '▙', '➭', '܀', 'ⓖ', '⇛', '▊', 
        '⇗', '̷', '⇱', '℅', 'ⓧ', '⚛', '̐', '̕', '⇌', '␀', '≌', 'ⓦ', '⊤', '̓', '☦', 'ⓕ', '▜', '➙', 'ⓨ', '⌨', '◮', '☷', 
        '◍', 'ⓚ', '≔', '⏩', '⍳', '℞', '┋', '˻', '▚', '≺', 'ْ', '▟', '➻', '̪', '⏪', '̉', '⎞', '┇', '⍟', '⇪', '▎', '⇦', '␝', 
        '⤷', '≖', '⟶', '♗', '̴', '♄', 'ͨ', '̈', '❜', '̡', '▛', '✁', '➩', 'ா', '˂', '↥', '⏎', '⎷', '̲', '➖', '↲', '⩵', '̗', '❢', 
        '≎', '⚔', '⇇', '̑', '⊿', '̖', '☍', '➹', '⥊', '⁁', '✢','∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', 
        'é', '&amp;','₹', 'á', '²', 'ế', '청', '하', '¨', '‘', '√', '\xa0', '高', '端', '大', '气', '上', '档', '次', '_', '½', 'π', '#', 
        '小', '鹿', '乱', '撞', '成', '语', 'ë', 'à', 'ç', '@', 'ü', 'č', 'ć', 'ž', 'đ', '°', 'द', 'े', 'श', '्', 'र', 'ो', 'ह', 
        'ि', 'प', 'स', 'थ', 'त', 'न', 'व', 'ा', 'ल', 'ं', '林', '彪', '€', '\u200b', '˚', 'ö', '~', '—', '越', '人', 'च', 'म', 'क', 
        'ु', 'य', 'ी', 'ê', 'ă', 'ễ', '∞', '抗', '日', '神', '剧', '，', '\uf02d', '–', 'ご', 'め', 'な', 'さ', 'い', 'す', 
        'み', 'ま', 'せ', 'ん', 'ó', 'è', '£', '¡', 'ś', '≤', '¿', 'λ', '魔', '法', '师', '）', 'ğ', 'ñ', 'ř', '그', '자', '식', '멀', 
        '쩡', '다', '인', '공', '호', '흡', '데', '혀', '밀', '어', '넣', '는', '거', '보', '니', 'ǒ', 'ú', '️', 'ش', 'ه', 'ا', 'د',
        'ة', 'ل', 'ت', 'َ', 'ع', 'م', 'ّ', 'ق', 'ِ', 'ف', 'ي', 'ب', 'ح', 'ْ', 'ث', '³', '饭', '可', '以', '吃', '话', '不', '讲', 
        '∈', 'ℝ', '爾', '汝', '文', '言', '∀', '禮', 'इ', 'ब', 'छ', 'ड', '़', 'ʒ', '有', '「', '寧', '錯', '殺', '一', '千', '絕', 
        '放', '過', '」', '之', '勢', '㏒', '㏑', 'ू', 'â', 'ω', 'ą', 'ō', '精', '杯', 'í', '生', '懸', '命', 'ਨ', 'ਾ', 'ਮ', 'ੁ', 
        '₁', '₂', 'ϵ', 'ä', 'к', 'ɾ', '\ufeff', 'ã', '©', '\x9d', 'ū', '™', '＝', 'ù', 'ɪ', 'ŋ', 'خ', 'ر', 'س', 'ن', 'ḵ', 'ā',
        'σ', '≡', '¹', '⊆', 'ı', '∆', 'μ', '卐', '¿', '∑', '≥', 'å', 'x₁','∆g','ⁿ','∘','▾','ψ', 'का', 'एक', 'को', 'लगाना', 'ß']

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",
                "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will",
                "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",
                "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
                "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling',
                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are',
                'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'bitcoin', 'narcissit': 'narcissist',
                'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',
                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 
                'electroneum':'bitcoin','nanodegree':'degree','hotstar':'star','dream11':'dream','ftre':'fire','tensorflow':'framework','unocoin':'bitcoin',
                'lnmiit':'limit','unacademy':'academy','altcoin':'bitcoin','altcoins':'bitcoin','litecoin':'bitcoin','coinbase':'bitcoin','cryptocurency':'cryptocurrency',
                'simpliv':'simple','quoras':'quora','schizoids':'psychopath','remainers':'remainder','twinflame':'soulmate','quorans':'quora','brexit':'demonetized',
                'cryptocoin':'bitcoin','blockchains':'blockchain','fiancee':'fiance','redmi':'smartphone','oneplus':'smartphone','qoura':'quora','deepmind':'framework','ryzen':'cpu','whattsapp':'whatsapp',
                'undertale':'adventure','zenfone':'smartphone','cryptocurencies':'cryptocurrencies','koinex':'bitcoin','zebpay':'bitcoin','binance':'bitcoin','whtsapp':'whatsapp',
                'reactjs':'framework','bittrex':'bitcoin','bitconnect':'bitcoin','bitfinex':'bitcoin','yourquote':'your quote','whyis':'why is','jiophone':'smartphone',
                'dogecoin':'bitcoin','onecoin':'bitcoin','poloniex':'bitcoin','7700k':'cpu','angular2':'framework','segwit2x':'bitcoin','hashflare':'bitcoin','940mx':'gpu',
                'openai':'framework','hashflare':'bitcoin','1050ti':'gpu','nearbuy':'near buy','freebitco':'bitcoin','antminer':'bitcoin','filecoin':'bitcoin','whatapp':'whatsapp',
                'empowr':'empower','1080ti':'gpu','crytocurrency':'cryptocurrency','8700k':'cpu','whatsaap':'whatsapp','g4560':'cpu','payymoney':'pay money',
                'fuckboys':'fuck boys','intenship':'internship','zcash':'bitcoin','demonatisation':'demonetization','narcicist':'narcissist','mastuburation':'masturbation',
                'trignometric':'trigonometric','cryptocurreny':'cryptocurrency','howdid':'how did','crytocurrencies':'cryptocurrencies','phycopath':'psychopath',
                'bytecoin':'bitcoin','possesiveness':'possessiveness','scollege':'college','humanties':'humanities','altacoin':'bitcoin','demonitised':'demonetized',
                'brasília':'brazilia','accolite':'accolyte','econimics':'economics','varrier':'warrier','quroa':'quora','statergy':'strategy','langague':'language',
                'splatoon':'game','7600k':'cpu','gate2018':'gate 2018','in2018':'in 2018','narcassist':'narcissist','jiocoin':'bitcoin','hnlu':'hulu','7300hq':'cpu',
                'weatern':'western','interledger':'blockchain','deplation':'deflation', 'cryptocurrencies':'cryptocurrency', 'bitcoin':'blockchain cryptocurrency'
               }

abbreviations = {
                "€" : "euro",
                "4ao" : "for adults only",
                "a.m" : "before midday",
                "a3" : "anytime anywhere anyplace",
                "aamof" : "as a matter of fact",
                "acct" : "account",
                "adih" : "another day in hell",
                "afaic" : "as far as i am concerned",
                "afaict" : "as far as i can tell",
                "afaik" : "as far as i know",
                "afair" : "as far as i remember",
                "afk" : "away from keyboard",
                "app" : "application",
                "approx" : "approximately",
                "apps" : "applications",
                "asap" : "as soon as possible",
                "asl" : "age, sex, location",
                "atk" : "at the keyboard",
                "ave." : "avenue",
                "aymm" : "are you my mother",
                "ayor" : "at your own risk", 
                "b&b" : "bed and breakfast",
                "b+b" : "bed and breakfast",
                "b.c" : "before christ",
                "b2b" : "business to business",
                "b2c" : "business to customer",
                "b4" : "before",
                "b4n" : "bye for now",
                "b@u" : "back at you",
                "bae" : "before anyone else",
                "bak" : "back at keyboard",
                "bbbg" : "bye bye be good",
                "bbc" : "british broadcasting corporation",
                "bbias" : "be back in a second",
                "bbl" : "be back later",
                "bbs" : "be back soon",
                "be4" : "before",
                "bfn" : "bye for now",
                "blvd" : "boulevard",
                "bout" : "about",
                "brb" : "be right back",
                "bros" : "brothers",
                "brt" : "be right there",
                "bsaaw" : "big smile and a wink",
                "btw" : "by the way",
                "bwl" : "bursting with laughter",
                "c/o" : "care of",
                "cet" : "central european time",
                "cf" : "compare",
                "cia" : "central intelligence agency",
                "csl" : "can not stop laughing",
                "cu" : "see you",
                "cul8r" : "see you later",
                "cv" : "curriculum vitae",
                "cwot" : "complete waste of time",
                "cya" : "see you",
                "cyt" : "see you tomorrow",
                "dae" : "does anyone else",
                "dbmib" : "do not bother me i am busy",
                "diy" : "do it yourself",
                "dm" : "direct message",
                "dwh" : "during work hours",
                "e123" : "easy as one two three",
                "eet" : "eastern european time",
                "eg" : "example",
                "embm" : "early morning business meeting",
                "encl" : "enclosed",
                "encl." : "enclosed",
                "etc" : "and so on",
                "faq" : "frequently asked questions",
                "fawc" : "for anyone who cares",
                "fb" : "facebook",
                "fc" : "fingers crossed",
                "fig" : "figure",
                "fimh" : "forever in my heart", 
                "ft." : "feet",
                "ft" : "featuring",
                "ftl" : "for the loss",
                "ftw" : "for the win",
                "fwiw" : "for what it is worth",
                "fyi" : "for your information",
                "g9" : "genius",
                "gahoy" : "get a hold of yourself",
                "gal" : "get a life",
                "gcse" : "general certificate of secondary education",
                "gfn" : "gone for now",
                "gg" : "good game",
                "gl" : "good luck",
                "glhf" : "good luck have fun",
                "gmt" : "greenwich mean time",
                "gmta" : "great minds think alike",
                "gn" : "good night",
                "g.o.a.t" : "greatest of all time",
                "goat" : "greatest of all time",
                "goi" : "get over it",
                "gps" : "global positioning system",
                "gr8" : "great",
                "gratz" : "congratulations",
                "gyal" : "girl",
                "h&c" : "hot and cold",
                "hp" : "horsepower",
                "hr" : "hour",
                "hrh" : "his royal highness",
                "ht" : "height",
                "ibrb" : "i will be right back",
                "ic" : "i see",
                "icq" : "i seek you",
                "icymi" : "in case you missed it",
                "idc" : "i do not care",
                "idgadf" : "i do not give a damn fuck",
                "idgaf" : "i do not give a fuck",
                "idk" : "i do not know",
                "ie" : "that is",
                "i.e" : "that is",
                "ifyp" : "i feel your pain",
                "IG" : "instagram",
                "iirc" : "if i remember correctly",
                "ilu" : "i love you",
                "ily" : "i love you",
                "imho" : "in my humble opinion",
                "imo" : "in my opinion",
                "imu" : "i miss you",
                "iow" : "in other words",
                "irl" : "in real life",
                "j4f" : "just for fun",
                "jic" : "just in case",
                "jk" : "just kidding",
                "jsyk" : "just so you know",
                "l8r" : "later",
                "lb" : "pound",
                "lbs" : "pounds",
                "ldr" : "long distance relationship",
                "lmao" : "laugh my ass off",
                "lmfao" : "laugh my fucking ass off",
                "lol" : "laughing out loud",
                "ltd" : "limited",
                "ltns" : "long time no see",
                "m8" : "mate",
                "mf" : "motherfucker",
                "mfs" : "motherfuckers",
                "mfw" : "my face when",
                "mofo" : "motherfucker",
                "mph" : "miles per hour",
                "mr" : "mister",
                "mrw" : "my reaction when",
                "ms" : "miss",
                "mte" : "my thoughts exactly",
                "nagi" : "not a good idea",
                "nbc" : "national broadcasting company",
                "nbd" : "not big deal",
                "nfs" : "not for sale",
                "ngl" : "not going to lie",
                "nhs" : "national health service",
                "nrn" : "no reply necessary",
                "nsfl" : "not safe for life",
                "nsfw" : "not safe for work",
                "nth" : "nice to have",
                "nvr" : "never",
                "nyc" : "new york city",
                "oc" : "original content",
                "og" : "original",
                "ohp" : "overhead projector",
                "oic" : "oh i see",
                "omdb" : "over my dead body",
                "omg" : "oh my god",
                "omw" : "on my way",
                "p.a" : "per annum",
                "p.m" : "after midday",
                "pm" : "prime minister",
                "poc" : "people of color",
                "pov" : "point of view",
                "pp" : "pages",
                "ppl" : "people",
                "prw" : "parents are watching",
                "ps" : "postscript",
                "pt" : "point",
                "ptb" : "please text back",
                "pto" : "please turn over",
                "qpsa" : "what happens",
                "ratchet" : "rude",
                "rbtl" : "read between the lines",
                "rlrt" : "real life retweet", 
                "rofl" : "rolling on the floor laughing",
                "roflol" : "rolling on the floor laughing out loud",
                "rotflmao" : "rolling on the floor laughing my ass off",
                "rt" : "retweet",
                "ruok" : "are you ok",
                "sfw" : "safe for work",
                "sk8" : "skate",
                "smh" : "shake my head",
                "sq" : "square",
                "srsly" : "seriously", 
                "ssdd" : "same stuff different day",
                "tbh" : "to be honest",
                "tbs" : "tablespooful",
                "tbsp" : "tablespooful",
                "tfw" : "that feeling when",
                "thks" : "thank you",
                "tho" : "though",
                "thx" : "thank you",
                "tia" : "thanks in advance",
                "til" : "today i learned",
                "tl;dr" : "too long i did not read",
                "tldr" : "too long i did not read",
                "tmb" : "tweet me back",
                "tntl" : "trying not to laugh",
                "ttyl" : "talk to you later",
                "u" : "you",
                "u2" : "you too",
                "u4e" : "yours for ever",
                "utc" : "coordinated universal time",
                "w/" : "with",
                "w/o" : "without",
                "w8" : "wait",
                "wassup" : "what is up",
                "wb" : "welcome back",
                "wtf" : "what the fuck",
                "wtg" : "way to go",
                "wtpa" : "where the party at",
                "wuf" : "where are you from",
                "wuzup" : "what is up",
                "wywh" : "wish you were here",
                "yd" : "yard",
                "ygtr" : "you got that right",
                "ynk" : "you never know",
                "zzz" : "sleeping bored and tired",
                "ain't": "am not",
                "aren't": "are not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "couldn't've": "could not have",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hadn't've": "had not have",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'd've": "he would have",
                "he'll": "he will",
                "he'll've": "he will have",
                "he's": "he is",
                "how'd": "how did",
                "how'd'y": "how do you",
                "how'll": "how will",
                "how's": "how does",
                "i'd": "i would",
                "i'd've": "i would have",
                "i'll": "i will",
                "i'll've": "i will have",
                "i'm": "i am",
                "i've": "i have",
                "isn't": "is not",
                "it'd": "it would",
                "it'd've": "it would have",
                "it'll": "it will",
                "it'll've": "it will have",
                "it's": "it is",
                "let's": "let us",
                "ma'am": "madam",
                "mayn't": "may not",
                "might've": "might have",
                "mightn't": "might not",
                "mightn't've": "might not have",
                "must've": "must have",
                "mustn't": "must not",
                "mustn't've": "must not have",
                "needn't": "need not",
                "needn't've": "need not have",
                "o'clock": "of the clock",
                "oughtn't": "ought not",
                "oughtn't've": "ought not have",
                "shan't": "shall not",
                "sha'n't": "shall not",
                "shan't've": "shall not have",
                "she'd": "she would",
                "she'd've": "she would have",
                "she'll": "she will",
                "she'll've": "she will have",
                "she's": "she is",
                "should've": "should have",
                "shouldn't": "should not",
                "shouldn't've": "should not have",
                "so've": "so have",
                "so's": "so is",
                "that'd": "that would",
                "that'd've": "that would have",
                "that's": "that is",
                "there'd": "there would",
                "there'd've": "there would have",
                "there's": "there is",
                "they'd": "they would",
                "they'd've": "they would have",
                "they'll": "they will",
                "they'll've": "they will have",
                "they're": "they are",
                "they've": "they have",
                "to've": "to have",
                "wasn't": "was not",
                "ur": "your",
                "n": "and",
                "won't": "would not",
                "dis": "this",
                "brng": "bring"
                }

# Feature Engineering

def get_charcounts(x):
	s = x.split()
	x = ''.join(s)
	return len(x)

def get_wordcounts(x):
	length = len(str(x).split())
	return length

def get_avg_wordlength(x):
	count = get_charcounts(x)/get_wordcounts(x)
	return count

def get_stopwords_counts(x):
	l = len([t for t in x.split() if t in stopwords])
	return l

def get_digit_counts(x):
	digits = re.findall(r'[0-9,.]+', x)
	return len(digits)

def get_uppercase_counts(x):
	return len([t for t in x.split() if t.isupper()])

def get_features(df, text_column):
  if type(df) == pd.core.frame.DataFrame:

    df['question_text'] = df[text_column].apply(lambda x: str(x))
    df['char_counts'] = df[text_column].apply(lambda x: get_charcounts(x))
    df['char_length'] =df[text_column].apply(lambda x: len(str(x)))
    df['word_counts'] = df[text_column].apply(lambda x: get_wordcounts(x))
    df['avg_wordlength'] = df[text_column].apply(lambda x: get_avg_wordlength(x))
    df['stopwords_counts'] = df[text_column].apply(lambda x: get_stopwords_counts(x))
    df['digits_counts'] = df[text_column].apply(lambda x: get_digit_counts(x))
    df['uppercase_counts'] = df[text_column].apply(lambda x: get_uppercase_counts(x))
    df['titlewords_count'] = df[text_column].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df['unique_words_count'] = df[text_column].apply(lambda x: len(set(str(x).split())))
    df['numbers_count'] = df[text_column].apply(lambda comment: sum(1 for c in comment if c.isdigit()))
    df['punctuations_count'] =df[text_column].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
  
  else:
     print('ERROR: This function takes only Pandas DataFrame')
		
  return df

# Text Cleaning

def remove_stopwords(x):
	return ' '.join([t for t in x.split() if t not in stopwords])	

#https://www.kaggle.com/canming/ensemble-mean-iii-64-36
def clean_tag(x):
    if '[math]' in x:
        x = re.sub('\[math\].*?math\]', 'math eqaution', x)
    if 'http' in x or 'www' in x:
        x = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', 'URL', x)
    return x

def remove_symbols_and_punctuations(x):
  x = str(x)
  for symbol in symbols_and_punctuations:
    if symbol in x:
      x = x.replace(symbol, ' ')
    return x

def get_spellcorrect(x):
  words = x.split()
  for i in range(0, len(words)):
    if mispell_dict.get(words[i]) is not None:
      words[i] = mispell_dict.get(words[i])
    elif mispell_dict.get(words[i].lower()) is not None:
      words[i] = mispell_dict.get(words[i].lower())
        
  words = " ".join(words)
  return words

def remove_digits(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def remove_abbreviations(x):
	# abbreviations = json.load(open('/content/drive/MyDrive/Quora Insincere Questions Classification/abbreviations.json'))
	if type(x) is str:
		for key in abbreviations:
			value = abbreviations[key]
			raw_text = r'\b' + key + r'\b'
			x = re.sub(raw_text, value, x)
		return x
	else:
		return x

def remove_html_tags(x):
	return BeautifulSoup(x, 'lxml').get_text().strip()
 
def remove_multiple_whitespace(x):
  return re.sub(' +', ' ', x)

def remove_non_alphanumeric(x):
  return re.sub('[\W_]+', ' ', x, flags=re.UNICODE)

def get_clean_text(x):
  x = str(x).lower().replace('\\', ' ').replace('_', ' ').replace('.', ' ')  
  x = clean_tag(x)
  x = remove_symbols_and_punctuations(x)
  x = get_spellcorrect(x)
  x = remove_stopwords(x)
  x = remove_abbreviations(x)
  return x

def preprocess(x):
  x = remove_multiple_whitespace(x)
  x = remove_symbols_and_punctuations(x)
  x = remove_abbreviations(x)
  x = get_spellcorrect(x)
  x = remove_html_tags(x)
  x = remove_non_alphanumeric(x)
  x = remove_digits(x)
  return x 

lemmatizer = WordNetLemmatizer()
def get_lemmatization(x):
  x = x.split()
  x = [lemmatizer.lemmatize(word) for word in x]
  x = ' '.join(x)
  return x

def get_vocab(df, text_column, verbose =  True):
  sentences = df[text_column].apply(lambda x: x.split()).values
  vocab = {}
  for sentence in tqdm(sentences, disable = (not verbose)):
      for word in sentence:
          try:
              vocab[word] += 1
          except KeyError:
              vocab[word] = 1
              
  return vocab

#https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings?scriptVersionId=7347748&cellId=13
# check coverage of embedding vs vocabulary of training data
def check_coverage(vocab,embeddings_index):
    a = {}
    # out of vocabulary (oov) 
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x