from capreolus.registry import ModuleBase, RegisterableModule, Dependency
from capreolus.utils.common import get_code_parser

class Tokenizer(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "tokenizer"


class AnseriniTokenizer(Tokenizer):
    name = "anserini"

    @staticmethod
    def config():
        keepstops = True
        stemmer = "none"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._tokenize = self._get_tokenize_fn()

    def _get_tokenize_fn(self):
        from jnius import autoclass

        stemmer, keepstops = self.cfg["stemmer"], self.cfg["keepstops"]
        emptyjchar = autoclass("org.apache.lucene.analysis.CharArraySet").EMPTY_SET
        Analyzer = autoclass("io.anserini.analysis.DefaultEnglishAnalyzer")
        analyzer = Analyzer.newStemmingInstance(stemmer, emptyjchar) if keepstops else Analyzer.newStemmingInstance(stemmer)
        tokenizefn = autoclass("io.anserini.analysis.AnalyzerUtils").analyze

        def _tokenize(sentence):
            return tokenizefn(analyzer, sentence).toArray()

        return _tokenize

    def tokenize(self, sentences):
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self._tokenize(sentences)

        return [self._tokenize(s) for s in sentences]


class CodeTokenizer(AnseriniTokenizer):
    """ A tokenizer specially handles camel and snake naming form on top of Anserini tokenization """
    name = "code"

    @staticmethod
    def config():
        keepstops = True
        stemmer = "none"
        remove_punc = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self._tokenize = self._get_tokenize_fn()

    def _get_tokenize_fn(self):
        """ add code tokenizer on top of the super() tokenizer """
        tokenizefn = super()._get_tokenize_fn()
        code_parser = get_code_parser(self.cfg["remove_punc"])

        def _tokenize(sentence):
            return tokenizefn(code_parser(sentence))

        return _tokenize
