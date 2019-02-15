from datetime import date
from dateutil.parser import parse
import gzip
import json
import re
from settings import get_config
import xml.etree.ElementTree as ET


def run(config):

    ENCODING = config['encoding']
    LOG_FILEPATH = config['extract_data_log_file']
    START_DATA_FILE_NUM = config['start_data_file_num']
    END_DATA_FILE_NUM = config['end_data_file_num']
    DATA_FILEPATH_TEMPLATE = config['data_filepath_template']
    JSON_FILEPATH_TEMPLATE = config['json_filepath_template']
    MEDLINE_CITATION_NODE_PATH = config['medline_citation_node_path']

    with open(LOG_FILEPATH, 'wt', encoding=ENCODING) as log_file: 
        for file_num in range(START_DATA_FILE_NUM, END_DATA_FILE_NUM + 1):
            data_filepath = DATA_FILEPATH_TEMPLATE.format(file_num)
            json_filepath = JSON_FILEPATH_TEMPLATE.format(file_num)
            with gzip.open(data_filepath, 'rt', encoding=ENCODING) as data_file, \
                 gzip.open(json_filepath, 'wt', encoding=ENCODING) as json_file:
                json_file_data = _get_json_file_data(data_file, MEDLINE_CITATION_NODE_PATH, log_file)
                json.dump(json_file_data, json_file, ensure_ascii=False, indent=4)
            log_file.flush()


def _article_is_relevant(article_metadata):
    pmid, title, _, journal_nlmid, pub_year, date_completed, citation_status, _ = article_metadata
    medline_or_pubmed = (citation_status == 'MEDLINE') or (citation_status == 'PubMed-not-MEDLINE')
    should_extract = pmid and \
                     title and \
                     journal_nlmid and \
                     pub_year and \
                     date_completed and \
                     medline_or_pubmed and \
    return should_extract


def _extract_article_metadata(medline_citation_node, log_file):

    citation_status = medline_citation_node.attrib['Status'].strip()

    pmid_node = medline_citation_node.find('PMID')
    pmid = pmid_node.text.strip()
    pmid = int(pmid)

    date_completed = None
    date_completed_node = medline_citation_node.find('DateCompleted')
    if date_completed_node is not None:
        date_completed_year = date_completed_node.find('Year').text.strip()
        date_completed_month = date_completed_node.find('Month').text.strip()
        date_completed_day = date_completed_node.find('Day').text.strip()
        date_completed = date(int(date_completed_year), int(date_completed_month), int(date_completed_day))
       
    journal_nlmid_node = medline_citation_node.find('MedlineJournalInfo/NlmUniqueID')
    journal_nlmid = journal_nlmid_node.text.strip() if journal_nlmid_node is not None else ''

    medlinedate_node = medline_citation_node.find('Article/Journal/JournalIssue/PubDate/MedlineDate')
    if medlinedate_node is not None:
        medlinedate_text = medlinedate_node.text.strip()
        pub_year = _extract_year_from_medlinedate(pmid, medlinedate_text, log_file)
    else:
        pub_year_node = medline_citation_node.find('Article/Journal/JournalIssue/PubDate/Year')
        pub_year = pub_year_node.text.strip()
        pub_year = int(pub_year)
    
    title = ''
    title_node = medline_citation_node.find('Article/ArticleTitle') 
    title = ET.tostring(title_node, encoding='unicode', method='text')
    if title is not None:
        title = title.strip()

    abstract = ''
    abstract_node = medline_citation_node.find('Article/Abstract')
    if abstract_node is not None:
        abstract_text_nodes = abstract_node.findall('AbstractText')
        for abstract_text_node in abstract_text_nodes:
            if 'Label' in abstract_text_node.attrib:
                if len(abstract) > 0:
                    abstract += ' '
                abstract += abstract_text_node.attrib['Label'].strip() + ': '
            abstract_text = ET.tostring(abstract_text_node, encoding='unicode', method='text')
            if abstract_text is not None:
                abstract += abstract_text.strip()

    comments_corrections_ref_types = []
    comments_corrections_list_node = medline_citation_node.find('CommentsCorrectionsList')
    if comments_corrections_list_node is not None:
        for comments_corrections_node in comments_corrections_list_node.findall('CommentsCorrections'):
            ref_type = comments_corrections_node.attrib['RefType'].strip()
            comments_corrections_ref_types.append(ref_type)

    return pmid, title, abstract, journal_nlmid, pub_year, date_completed, citation_status, comments_corrections_ref_types


def _extract_year_from_medlinedate(pmid, medlinedate_text, log_file):
    pub_year = medlinedate_text[:4]
    try:
        pub_year = int(pub_year)
        #log_file.write('First4,{},{}\n'.format(pub_year, pmid))
    except ValueError:
        match = re.search(r'\d{4}', medlinedate_text)
        if match:
            pub_year = match.group(0)
            pub_year = int(pub_year)
            log_file.write('YearRe,{},{}\n'.format(pub_year, pmid))
        else:
            try:
                pub_year = parse(medlinedate_text, fuzzy=True).date().year
                log_file.write('Fuzzy,{},{}\n'.format(pub_year, pmid))
            except ValueError:
                pub_year = None
                log_file.write('Invalid,{},{}\n'.format(medlinedate_text, pmid))
    if pub_year:
        if 1500 >= pub_year <= 2100:
            log_file.write('OutOfRange,{},{}\n'.format(pub_year, pmid))
    return pub_year


def _get_json_data_for_article(article_metadata):
    pmid, title, abstract, journal_nlmid, pub_year, date_completed, citation_status, comments_corrections_ref_types = article_metadata
    ref_types = list(set(comments_corrections_ref_types))
    article = { 'pmid': pmid, 
                'title': title, 
                'abstract': abstract, 
                'pub_year': pub_year,
                'date_completed': date_completed.isoformat(),
                'journal_nlmid': journal_nlmid, 
                'is_indexed': citation_status == 'MEDLINE',
                'ref_types': ref_types,
                }
    return article


def _get_json_file_data(data_file, medline_citation_node_path, log_file):
    articles = []
    root_node = ET.parse(data_file)
    for medline_citation_node in root_node.findall(medline_citation_node_path):
         article_metadata = _extract_article_metadata(medline_citation_node, log_file)
         if _article_is_relevant(article_metadata):
            article = _get_json_data_for_article(article_metadata)
            articles.append(article)
    return { 'articles': articles }


if __name__ == '__main__':
    config = get_config()
    run(config)  