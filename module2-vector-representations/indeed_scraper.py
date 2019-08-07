import urllib3

from bs4 import BeautifulSoup

URL = 'http://www.indeed.com/jobs?q=data%20scientist&l=Seattle,%20WA'


class IndeedScraper(object):
    """Get the long descriptions of Indeed.com job listings.

    You supply the base URL.

    Go to Indeed.com and enter the job-search term and city.
    Right click on the 'Find jobs' button and copy the link.
    Paste the link as an argument when instantiating the scraper.

    list_o_descriptions = IndeedScraper('http://www.indeed.com/jobs?q=data%20scientist&l=Seattle,%20WA').get_descriptions()

    :param url: string URL of job and city to search for.
    :param pages: int number of pages to scrape.
    :returns list: strings of job descriptions.

    """

    def __init__(self, url: str, pages: int):
        self.url = url
        self.http = urllib3.PoolManager()
        self.pages = pages

    def find_long_urls(self, soup):
        urls = []
        for div in soup.find_all(name='div', attrs={'class': 'row'}):
            for a in div.find_all(name='a', attrs={'class': 'jobtitle turnstileLink'}):
                urls.append(a['href'])
        return urls

    def get_next_pages(self):
        return [self.url] + [self.url + str(x) + '0' for x in range(1, self.pages)]

    def get_descriptions(self):
        descriptions = []
        for base_url in self.get_next_pages():
            request = self.http.request('GET',
                                        base_url)
            base_soup = BeautifulSoup(request.data)

            for url in self.find_long_urls(base_soup):
                the_url = "http://www.indeed.com/" + url

                req = self.http.request('GET', the_url,
                                        headers={'User-Agent': 'opera'},
                                        retries=urllib3.Retry(connect=500, read=2, redirect=50))

                soup = BeautifulSoup(req.data, 'html.parser')
                description = soup.find(name='div', attrs={'id': 'jobDescriptionText'})
                descriptions.append(description.text)

        return descriptions


get_em = IndeedScraper(URL, 10).get_descriptions()
print(get_em)
