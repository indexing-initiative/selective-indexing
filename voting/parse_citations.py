import lxml.etree as le
import pandas as pd


class CitationParser():
    """
    Parse citations from the training data
    """

    def get_abstract(row):
        """
        get the abstracts from the citation
        """

        root = le.fromstring(row['citation'])
        le.cleanup_namespaces(root)
        abstract = root.find('.//Abstract')
        if abstract is not None:
            data = pd.Series([
                le.tostring(abstract, encoding='unicode', method='text'),
                row['label'],
                row['pmid'],
                row['pub_date'],
                row['journal']],
                index=['abstract', 'label', 'pmid', 'pub_date', 'journal'])
        # make sure to return none if there is no abstract.
        else:
            data = pd.Series([
                "None",
                row['label'],
                row['pmid'],
                row['pub_date'],
                row['journal']],
                index=['abstract', 'label', 'pmid', 'pub_date', 'journal'])

        return data


    def get_title(row):
        """
        Get the title from the citation
        """

        root = le.fromstring(row['citation'])
        title = root.find('.//ArticleTitle')
        if title is not None:
            title_string = le.tostring(title, encoding='unicode', method='text')
        else:
            title_string = "None"

        return title_string

    def filter_citations(row):
        """
        Remove some citations from the dataset,
        based on the RefType attribute in the
        CommentsCorrectionsList
        """

        attribute_list = ['CommentOn',
                          'ErratumFor',
                          'ExpressionOfConcernFor',
                          'RepublishedFrom',
                          'RetractionOf',
                          'UpdateOf',
                          'PartialRetractionOf']

        root = le.fromstring(row['citation'])
        comments = root.find('.//CommentsCorrectionsList')

        data = pd.Series([
            row['citation'],
            row['label'],
            row['pmid'],
            row['pub_date'],
            row['journal']],
            index=['citation', 'label', 'pmid', 'pub_date', 'journal'])

        if comments is not None:
            for comment in comments:
                if comment.attrib['RefType'] in attribute_list:
                    data = pd.Series([
                        None,
                        row['label'],
                        row['pmid'],
                        row['pub_date'],
                        row['journal']],
                        index=['citation', 'label', 'pmid', 'pub_date', 'journal'])
                    break

        return data

    def clean_citation(row):
        """
        Remove the stuff the indexers add from the citations.
        This tests how well the raw citation xml can be classified.
        """

        tag_list = [
            'ChemicalList',
            'MeshHeadingList',
            'PublicationTypeList',
            'GrantList',
            'DataBankList',
            'CommentsCorrectionsList',
            'PubmedData',
            'PMID',
            'DateCompleted',
            'DateRevised',
            'Pagination',
            'MedlineJournalInfo',
            'CitationSubset'
            ]

        # Remove the easy to get tags
        root = le.fromstring(row['citation'])
        for tag in tag_list:
            superfluous_tag = root.find(".//{}".format(tag))
            if superfluous_tag is not None:
                superfluous_tag.getparent().remove(superfluous_tag)

        #print(le.tostring(root, encoding='unicode', method='xml'))

        # Remove the medline_citations tag which contains label info
        citation_status = root.find('.//MedlineCitation')
        citation_status.attrib.clear()

        data = pd.Series([
            le.tostring(root, encoding='unicode', method='xml'),
            row['label']],
            index=['abstract', 'label'])

        return data

    def get_affiliations(row):
        """
        Get just the affiliation information and not
        everything out of the authors list as the function
        above does
        """

        root = le.fromstring(row['citation'])
        affiliations = root.findall('.//AffiliationInfo')

        if affiliations is not None:
            affiliation_list = [le.tostring(affiliation, encoding='unicode', method='text') for affiliation in affiliations]
            affiliation_string = " ".join(affiliation_list)
        else:
            affiliation_string = "None"

        return affiliation_string
