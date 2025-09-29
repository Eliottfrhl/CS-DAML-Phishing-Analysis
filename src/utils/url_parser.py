import wordninja
import tldextract

class URLParser:
    @staticmethod
    def parse(url: str) -> list[str]:
        domain = tldextract.extract(url).domain
        return wordninja.split(domain) if domain else []