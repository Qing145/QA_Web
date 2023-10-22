import spacy

nlp = spacy.load("en_core_web_sm")
# doc = nlp("Autonomous cars shift insurance liability toward manufacturers")

while True:
    input_question = input("plz input your question: ")

    doc = nlp(input_question)

    '''
    root：中心词，通常是动词
    dobj：直接宾语（direct object）
    nsubj：名词性主语（nominal subject）
    prep：介词
    pobj：介词宾语
    conj: 连词
    cc：连词
    compound：复合词
    advmod：状语
    det：限定词
    amod：形容词修饰语
    '''
    # 名词块
    # for chunk in doc.noun_chunks:
    #     print(chunk.text, chunk.root.text, chunk.root.dep_,
    #             chunk.root.head.text)

    nsubj = ""
    ans = False  # 用来标记是否回答出了问题
    for chunk in doc.noun_chunks:
        print(chunk)
        if chunk.root.dep_ == "nsubj":
            nsubj = chunk.text
        if chunk.root.dep_ == "dobj" or chunk.root.dep_ == "pobj":
            '''
            dobj：直接宾语  一般为 who teach big data 这种句式, 其 relation 一般为动词, 需要由 chunk.root.head.text 提取
            pobj：介词宾语  一般为 what'a the teacher of big data 这种句式, 其 relation 一般为 nsubj
            '''
            if chunk.root.dep_ == "dobj":
                print("relation is : " + chunk.root.head.text)
                print("head entity is : " + chunk.text)
            else:
                print("relation is : " + nsubj)
                print("head entity is : " + chunk.text)
            ans = True
            break

    if not ans:
        # 在此解决 what/where/... 等问题
        if input_question.startswith("where is"):
            print("relation is : location")
            print("head entity is : " + nsubj)



    '''
    已知问题:
    1. java 这种词识别不了  -->  通过大写可以解决
    2. where is hospital 这种句子, 只能识别出一个 hospital - nsubj
    3. what is hospital 这种句子, 只能识别出一个 What - nsubj
    4. comp 这种词大写还是识别不了 COMP attr
    '''