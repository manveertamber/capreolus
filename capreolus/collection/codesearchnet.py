import json
import pickle
from zipfile import ZipFile
from collections import defaultdict, OrderedDict

from tqdm import tqdm
from nirtools.text import preprocess

from capreolus import ConfigOption, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import document_to_trectxt

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class CodeSearchNet(Collection):
    """CodeSearchNet Corpus. [1]

       [1] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. arXiv 2019.
    """

    module_name = "codesearchnet"
    url = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
    collection_type = "TrecCollection"  # TODO: any other supported type?
    generator_type = "DefaultLuceneDocumentGenerator"

    config_spec = [
        ConfigOption("lang", "ruby", "CSN language dataset to use"),
        ConfigOption("tokenizecode", True, "Whether to tokenize camelCase and snake_case in docstring and code"),
        ConfigOption("removekeywords", True, "Whether to remove reserved words in each programming language"),
        ConfigOption("removeunichar", True, "Whether to remove single-letter character"),
    ]

    @property
    def docmap(self):
        if not hasattr(self, "_docmap"):
            self.download_if_missing()
        return self._docmap

    @staticmethod
    def process_text(sent, remove_keywords=True, tokenize_code=True, remove_unichar=True, lang=None):
        sents = OrderedDict()

        sent = " ".join(sent.split())  # remove consecutive whitespace, \t, \n
        sents["raw"] = sent

        if remove_keywords:
            reserved_words = preprocess.get_lang_reserved_words(lang)  # it may raise ValueError
            sent = " ".join([word for word in sent.split() if word not in reserved_words])
            sents["no_reserved_word"] = sent

        sent = sent.lower()  # should not lowercase the sentence before since there might be uppercase in the reserved words
        if tokenize_code:
            sent = preprocess.code_tokenize(sent, return_str=True)
            sents["code_tokenized"] = sent

        sent = preprocess.remove_non_alphabet(sent, return_str=True)
        sents["no_nonalphabet"] = sent

        if remove_unichar:
            sent = preprocess.remove_unicharacter(sent, return_str=True)
            sents["no_unichar"] = sent

        sents["final"] = sent
        return sents

    def download_raw(self):
        tmp_dir = self.get_cache_path() / "tmp"
        zip_file = "%s.zip" % self.config["lang"]

        zip_path = tmp_dir / zip_file
        lang_url = f"{self.url}/{zip_file}"

        if zip_path.exists():
            logger.info(f"{zip_file} already exist under directory {tmp_dir}, skip downloaded")
        else:
            tmp_dir.mkdir(exist_ok=True, parents=True)
            download_file(lang_url, zip_path)

        with ZipFile(zip_path, "r") as zipobj:
            zipobj.extractall(tmp_dir)

        return tmp_dir

    def download_if_missing(self):
        cachedir = self.get_cache_path()
        tmp_dir, document_dir = cachedir / "tmp", cachedir / "documents"
        coll_filename = document_dir / ("csn-%s-collection.txt" % self.config["lang"])
        docmap_filename = tmp_dir / "docmap.json"

        if coll_filename.exists() and docmap_filename.exists():
            self._docmap = json.load(open(docmap_filename, "r"))
            return document_dir

        document_dir.mkdir(exist_ok=True, parents=True)
        raw_dir = self.download_raw()

        pkl_path = raw_dir / (self.config["lang"] + "_dedupe_definitions_v2.pkl")
        self._docmap = self.parse_pkl(pkl_path, coll_filename)
        json.dump(self._docmap, open(docmap_filename, "w", encoding="utf-8"))
        return document_dir

    def parse_pkl(self, pkl_path, trec_path):
        """
        prepare trec-format collection file and prepare document2id mapping
        :param pkl_path:
        :param trec_path:
        :return:
        """
        lang = self.config["lang"]
        with open(pkl_path, "rb") as f:
            codes = pickle.load(f)

        docmap = defaultdict(dict)
        fout = open(trec_path, "w", encoding="utf-8")
        for id, code in tqdm(enumerate(codes), desc=f"Preparing the {lang} collection file"):
            docno = f"{lang}-FUNCTION-{id}"
            docs = self.process_text(
                " ".join(code["function_tokens"]),
                lang=lang,
                remove_keywords=self.config["removekeywords"],
                tokenize_code=self.config["tokenizecode"],
                remove_unichar=self.config["removeunichar"])
            raw_doc, final_doc = docs["raw"], docs["final"]
            fout.write(document_to_trectxt(docno, final_doc))

            url = code["url"]
            if raw_doc in docmap.get(url, {}):
                logger.error(f"duplicate code: current code snippet {code['url']} duplicate the document "
                               f"{docmap[url][raw_doc]}}")
                # raise ValueError()
            elif url in docmap:
                docmap[url][raw_doc] = docno

        # remove the code_tokens for the unique url-docid mapping
        for url, docids in tqdm(docmap.items(), desc=f"Compressing the {lang} docid_map"):
            docmap[url] = list(docids.values()) if len(docids) == 1 else docids  # {code_tokens: docid} -> [docid]
        fout.close()
        return docmap
