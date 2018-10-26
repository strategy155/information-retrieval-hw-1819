import scrapy


COUNTER = 2000

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    counter = 0

    def check_ip(self, response):
        pub_ip = response.xpath('//body/text()').re(
            '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')[0]
        print("My public IP is: " + pub_ip)


    def start_requests(self):
        yield scrapy.Request('http://checkip.dyndns.org/', callback=self.check_ip)

        url = "https://www.avito.ru/moskva/transport"
        yield scrapy.Request(url=url, callback=self.parse_first)

    def parse_first(self, response):
        categories = response.css('a[class="js-catalog-counts__link"]::attr(href)').extract()
        for category in categories:
            yield scrapy.Request(response.urljoin(category), self.parse_category)

    def parse_category(self, response):
        commercials = response.css('a[class="item-description-title-link"]::attr(href)').extract()
        for commercial in commercials:
            self.counter+=1
            yield scrapy.Request(response.urljoin(commercial), self.parse_commercial)
        if self.counter < COUNTER:
            print(self.counter)
            next_page = response.css('a[class="pagination-page js-pagination-next"]::attr(href)').extract()
            if next_page:
                yield scrapy.Request(response.urljoin(next_page[0]), callback=self.parse_category)
        else:
            self.counter = 0
            return None




    def parse_commercial(self, response):
        filename = response.url.split('/')[-1]
        title = 'title: '
        price = 'price: '
        address = "address: "
        metadata = "meta: "
        additional_info = "info: "
        text = 'text: '
        if response.css('span[class="title-info-title-text"]::text').extract():
            title += response.css('span[class="title-info-title-text"]::text').extract()[0]
        if response.css('span[class="js-item-price"]::text').extract():
            price += response.css('span[class="js-item-price"]::text').extract()[0]
        if response.css('div[class="item-map-location"] *::text').extract():
            address += " ".join(response.css('div[class="item-map-location"] *::text').extract())
        if response.css('div[class="title-info-metadata-item"]::text').extract():
            metadata += response.css('div[class="title-info-metadata-item"]::text').extract()[0]
        if response.css('div[class="item-params"] *::text').extract():
            additional_info += " ".join(response.css('div[class="item-params"] *::text').extract())
        if  response.css('div[class="item-description-text"] *::text').extract():
            text +=  " ".join(response.css('div[class="item-description-text"] *::text').extract())
        with open('D:\\work\\ir\\corp\\' + filename + '.txt', 'w', encoding='utf-8') as commercial_file:
            commercial_file.write('\n'.join([title, price, address, metadata, additional_info, text]))

