"""URLs and content hashes for reviewed article passages."""

from hashlib import sha256

LOYALTY = "article:ef04f756-44bf-478a-b022-51fae04ecee5"
RESALE = "article:000ecbf2-ccb2-4305-82e2-f3ed7ce1b22c"
PRINTER = "article:33a3a4ca-ab5e-4e58-bbeb-62afae4de0ba"
FEEDBACK = "article:0d63697c-8fe9-4b4b-9a66-7cad43aab1ea"

SALON_RETURN = "article:bfc159df-4a6a-4ec1-9343-9920f1654c92"
SERVICE_QUALITY = "article:6ae980f7-41d2-4c8f-87f7-3348be389e4d"
LOCAL_DELIVERY = "article:4177c427-c698-4985-a5a2-e18e356bdae5"
PEST_ANALYSIS = "article:e07e384c-6f11-47c3-b2d6-dfd82e97f730"
CUSTOMER_CONFLICT = "article:d0ce5d7a-655a-49dc-a99e-ca014fefc469"
REMOTE_TEAM = "article:9e4e421f-bc27-4bac-aa7a-bc527e7b6539"
HIRING = "article:ce10a90f-8257-4ac9-9b86-e586d81c5c6c"
PRODUCT_SKU = "article:f3d25c33-90cc-47ab-8d55-c67d9241b049"
REPEAT_SALES = "article:33eae91e-6594-43b9-862a-8869c0e25f81"

REVIEWED = {
    SALON_RETURN: (
        "https://kurs.alfabank.ru/articles/kak-vernut-klientov-v-salon-krasoty/",
        {
            (9,): "2a26d73ee61dcdb4665773a6dc2fbae7a94aa91f712724b35ec87105db085dac",
            (13, 14): "3345d8f613e66708a8431ef9a2809f7c193512648afd0f019a12d56094bc444c",
        },
    ),
    SERVICE_QUALITY: (
        "https://kurs.alfabank.ru/articles/klientskij-servis-kak-uluchshit-kachestvo-obsluzhivaniya/",
        {
            (36, 37): "064c39b7f64972937aa0bafc54fe20502e972edae4bd082ef172afab9d3c5a1d",
            (38, 39): "6fac90ae8368e74ee8ba161a9aa33d53329b402f13f7977e456ae0b83e64ee92",
        },
    ),
    LOYALTY: (
        "https://kurs.alfabank.ru/articles/kak-sozdat-programmu-loyalnosti-dlya-klientov-salona-krasoty/",
        {
            (12,): "8447787a7c9e48ff2a9491df0a7c2d7d9fe04470e461a17e0c4642fc0efc910d",
            (13,): "6dba4a2a45eb2a7bd352744af58375a09208104696459758bfe18948c94a6dd6",
            (20, 21): "24a4f71f366dd89818bee8813f9a1b18e4540f3e8100e4832d6a25cab9972ed8",
        },
    ),
    RESALE: (
        "https://kurs.alfabank.ru/articles/chto-takoe-resejl-i-kak-nachat-zanimatsya-im-s-nulya/",
        {(32, 33, 34): "a277760a9c55c44300cf3c6d7876c31eb124c762542af5e9aba2c84cef23abc2"},
    ),
    PRINTER: (
        "https://kurs.alfabank.ru/articles/kak-vybrat-printer-ehtiketok-dlya-marketplejsov/",
        {
            (9, 10, 11, 12, 13): "055ee0357f0a7b579a195ff771673e5a275e5a0d4deb28005b2db5437311ea05",
            (31, 32, 33, 34): "ba0c8c07b22cabc8a19e6de98e464a886296ee22f30badb1870aa77a3a212c3d",
        },
    ),
    FEEDBACK: (
        "https://kurs.alfabank.ru/articles/itogovaya-obratnaya-svyaz-instrukciya-dlya-rukovoditelya/",
        {
            (23,): "73553f191b7f999b2b821b2cba639bda02a065b549d06c63fc27f62b7bff024c",
            (30,): "3727dad23e42fbd1c232fd7cbe931dd850c5db5f8ad18dcc75485fd2eef9489a",
        },
    ),
    LOCAL_DELIVERY: (
        "https://kurs.alfabank.ru/articles/giperlokalnaya-dostavka-kak-rabotaet-i-kogda-vygodna-predprinimatelyu/",
        {
            (14,): "67e224d6f713d82ffda639f715c271fa5b20a8786eb4eae7e4e3df46f1158677",
            (15, 16, 17, 18): "2f6a06f2cde8c8d33250e875e0a796d6523228d996e76f36c01fc20470c066e3",
        },
    ),
    PEST_ANALYSIS: (
        "https://kurs.alfabank.ru/articles/chto-takoe-pest-analiz-i-kak-ego-provesti/",
        {
            (7, 8, 9): "981eefda1410a1a6189a82a3d1d47aac27744a3f6a41d9f6217e17f2fb49834b",
            (18, 19): "38adffb90c248ddc6c5715fc6a25f42e85421fd58a341c190ef2940540b636ae",
        },
    ),
    CUSTOMER_CONFLICT: (
        "https://kurs.alfabank.ru/articles/tochka-kipeniya-kak-vesti-sebya-v-konflikte-s-klientom/",
        {
            (
                12,
                13,
                14,
                15,
                16,
                17,
            ): "14f1123bdc230f9a019ad941883c0d0989071d26081e4b6c563e05613163fc1b"
        },
    ),
    REMOTE_TEAM: (
        "https://kurs.alfabank.ru/articles/raspredelyonnaya-komanda-chto-takoe-kak-sobrat-i-kak-upravlyat/",
        {
            (20, 21): "6d96d516cf8bccf1e09cdf84419c9b3edd55a1d25d30f294290d00b732119220",
            (24, 25, 26): "96244ab88949f58d4e7d2395d76b9bcb1ab0b3619efe8f56f81172df3c14f345",
            (32, 33): "55f9bab165246994b0a6855d9f21f3243f9ba872a7a78f64e1957521ba000ec5",
            (40,): "3fc6d3b50cbc6970e4c8d67bb799c2b3124a0218158e195b5c57b7a1b1395a78",
        },
    ),
    HIRING: (
        "https://kurs.alfabank.ru/articles/kak-privlech-sotrudnikov-v-vashu-kompaniyu/",
        {
            (7,): "a94ce488b4d9668744a96687e9677542d70f8dc8bf8b441d7d4419d983ace747",
            (8,): "2625202c7f7a0f227236970cc7997cf35fd5ec2e051e18dccf0bd9fc0c90c326",
            (10,): "2692c1d255a23c371c7196e1406f427f87384c0e80345e201f0d85b1684761ef",
        },
    ),
    PRODUCT_SKU: (
        "https://kurs.alfabank.ru/articles/kak-pridumat-artikul-i-prisvoit-ego-tovaru/",
        {
            (3, 4): "2514aaa1b1c1ef205af819300a5e75ea97781fb50e0a1060208bf87687f9412a",
            (7,): "cd7f23ee1bb2d05821b8ed86c1f3a0602dbac7180b8d5acf3f4a30cb53fada78",
            (13,): "63b8b6cd0db7e0f81f1f321008c17d7a21b5bad8a398dba5664fa02a0c1a7181",
        },
    ),
    REPEAT_SALES: (
        "https://kurs.alfabank.ru/articles/kak-uvelichit-povtornye-prodazhi/",
        {
            (9, 10): "610642b126504353e7b1e01dcb491564bef13d0755db7d32e24c09f12f84c568",
            (
                34,
                35,
                36,
                37,
                38,
            ): "c96190baf77dab6af1f408c1eb01073218930ad90d3d99ffdef8d0fdeae89ab5",
            (39, 40, 41, 42): "22abfe5fc48e939b333496307b278828513f5b5ca8e1967d2c8b5cc1715d9d2b",
        },
    ),
}

# Zero-based paragraph groups in hash-verified passages. Keep the reason about
# print durability together with the recommendation about long storage.
PARAGRAPH_GROUPS = {
    (PRINTER, (9, 10, 11, 12, 13)): ((1, 2),),
    # Keep both methods in the comparison and the scales beside the scoring rule.
    (PEST_ANALYSIS, (7, 8, 9)): ((0, 1),),
    (PEST_ANALYSIS, (18, 19)): ((0, 1),),
}

# Search metadata for terminology used in questions. These strings are not evidence.
SEARCH_TERMS = {
    (
        REMOTE_TEAM,
        (20, 21),
    ): "Удалённая работа: после созвона сохранить решение письменно, журнал договорённостей",
    (
        REPEAT_SALES,
        (9, 10),
    ): "LTV покупателя: жизненная ценность, доход от клиента за весь срок взаимодействия",
}


def approved_passages(chunks):
    expected = {(doc, orders) for doc, (_, hashes) in REVIEWED.items() for orders in hashes}
    matched = set()
    approved = []
    for chunk in chunks:
        document_id = chunk["document_id"]
        orders = tuple(chunk["source_block_orders"])
        key = document_id, orders
        if key not in expected:
            continue
        url, hashes = REVIEWED[document_id]
        if sha256(chunk["text"].encode("utf-8")).hexdigest() != hashes[orders]:
            continue
        if key in matched:
            raise ValueError(f"Проверенный фрагмент продублирован: {document_id}, блоки {orders}")
        matched.add(key)
        # Build search input from the verified body and reviewed keywords.
        search_fields = [chunk["title"], *chunk["section"]]
        if terms := SEARCH_TERMS.get(key):
            search_fields.append(terms)
        search_header = " / ".join(search_fields)
        approved.append(
            {
                **chunk,
                "url": url,
                "embedding_text": f"passage: {search_header}\n\n{chunk['text'].strip()}",
            }
        )
    if matched != expected:
        raise ValueError(
            "Проверенные фрагменты отсутствуют или изменились. Нужна повторная сверка."
        )
    return approved
