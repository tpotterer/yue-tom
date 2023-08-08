import copy
from bs4 import BeautifulSoup
import re
import dateparser

quarter_regex = re.compile(f"Q[1-4]\s+20[0-2][0-9]")
brackets_regex = re.compile(f"\([^\)]+:[^\)]+\)")
brackets_regex_old = re.compile(f"\([^\)]+\)")
old_q_regex_1 = re.compile("F[1-4]Q\s?[0-2][0-9]") # F1Q 12 or F1Q12
old_q_regex_2 = re.compile("F[1-4]Q\s?20[0-2][0-9]") # F1 Q12
old_q_regex_3 = re.compile("Q[1-4]")
year_regex = re.compile("20[0-2][0-9]")
date_regex = re.compile(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4},?\s+\d{1,2}:\d{2}\s+(?:AM|PM)\s+ET)")
qa_regex = re.compile(r'question-and-answer\s?session\s?', re.IGNORECASE)

def get_quarter(title):
    for regex in [quarter_regex, old_q_regex_1, old_q_regex_2]:
        quarters = regex.findall(title)
        if len(quarters) == 1:
            return quarters[0]
    
    quarters = old_q_regex_3.findall(title)
    year = year_regex.findall(title)
    if(len(quarters) == 1 and len(year) == 1):
        return f"{quarters[0]} {year[0]}"

    if(len(quarters) == 0):
        raise Exception(f"No quarter found in title: {title}")
    if(len(quarters) > 1):
        raise Exception(f"Multiple quarters found in title: {title}")
    raise Exception(f"Unknown error in get_quarter: {title}")

def get_content(soup, filename):
    content = copy.copy(soup.find("div", class_='ks-mH'))
    if(content):
        return content
    
    content = copy.copy(soup.find("div", class_='lm-ls'))
    if(not content):
        raise Exception(f"Could not retrieve content from: {filename}")
    
    return content

def get_ticker_exchange(content, title):
    ticker_exchange = copy.copy(content.find("span", class_="ticker-hover-wrapper"))
    if(ticker_exchange):
        text = ticker_exchange.get_text()
        if(len(text) < 12):
            return text
    
    ticker_exchange = brackets_regex.findall(title)
    if(len(ticker_exchange)):
        return ticker_exchange[0][1:-1]
    
    ticker_exchange = brackets_regex_old.findall(title)
    if(len(ticker_exchange)):
        return ticker_exchange[0][1:-1]
    raise Exception(f"Could not find exchange in title: {title}")

def get_header(soup, filename):
    header = copy.copy(soup.find_all("h1"))
    if(not header):
        raise Exception(f"Could not find header in {filename}")
    return header[0].get_text()

def get_date(header, soup):
    dates = date_regex.findall(header)
    if(len(dates) == 0):
        raise Exception(f"Could not find date in {header}")
        
    clean_dates = [i for i in [str(dateparser.parse(date))[:10] for date in dates] if i !='None']
    if(not len(clean_dates) or clean_dates[0] < '2000'):
        date = copy.copy(soup.find("span", class_='rD-UA')).get_text()
        return str(dateparser.parse(date))[:10]
        
    return clean_dates[0]

def process_transcript(content):
    try:
        transcript_lines = [p for p in content.find_all("p")]
        
        company_roles = ['executives', 'company participants', 'corporate participants']
        other_roles = ['analysts', 'conference call participants']

        curr_line_speaker = "Unknown"
        curr_utterance = []
        lines = []
        is_presentation = True
        for line in transcript_lines: 
            # if line contains strong or span tag
            if(line.find("strong")):
                if(len(curr_utterance) > 0):
                    lines += [(curr_line_speaker, curr_utterance, is_presentation)]
                elif(re.match(qa_regex, curr_line_speaker)):
                    is_presentation = False
                curr_line_speaker = line.get_text()
                curr_utterance = []
            else:
                curr_utterance += [line.get_text()]
        
        header = lines[0]
        company_participants = lines[1]
        other_participants = lines[2]
        if(company_participants[0].lower().strip() not in company_roles):
            raise Exception(f"Could not find company participants in {company_participants[0]}")
        if(other_participants[0].lower().strip() not in other_roles):
            raise Exception(f"Could not find other participants in {other_participants[0]}")
        if(is_presentation):
            raise Exception(f"Could not find Q&A section")

        
        company_participants = company_participants[1]
        other_participants = other_participants[1]
        
        lines = lines[3:]
        return company_participants, other_participants, lines
    except Exception as e:
        raise Exception(f"Could not process transcript: {e}")

def extract_data(file_path):
    try:
        html = open(file_path, encoding='utf-8').read()
        soup = BeautifulSoup(html, 'html.parser')
        
        content = get_content(soup, file_path)
                
        company_participants, other_participants, transcript = process_transcript(copy.copy(content))
    
        header = get_header(soup, file_path)
        subtitle = copy.copy(soup.find("span", class_='rD-UA'))
        
        title = '\n'.join([p.get_text() for p in copy.copy(content.find_all("p")[:5])])
        
        ticker_exchange = get_ticker_exchange(content, title)
        
        quarter = get_quarter(title)
        
        date = get_date(title, soup)
        
        assert isinstance(quarter, str)
        assert isinstance(date, str)
        assert isinstance(title, str)
        assert isinstance(quarter, str)
        assert isinstance(transcript, list)
    
        return {
            'quarter': quarter,
            'date': date,
            'title': title,
            'ticker_exchange': ticker_exchange,
            'transcript': transcript,
            'company_participants': company_participants,
            'other_participants': other_participants,
            'file_name': file_path.split('/')[-1],
            'has_error': False,
        }
    except Exception as e:
        return {
            'has_error': True,
            'error': str(e),
        }