import pandas as pd
import requests
import numpy as np


def NotUnitcode(x):
    if x[:2] != 'U+':
        return np.nan
    else:
        return x
    
def category(df):
    df["category"] = 0
    #가족
    for i in range(493,519):
        df.loc[i, 'category'] = '여성/가족'
    #여성
    df.loc[237, 'category'] = '여성/가족'
    df.loc[238, 'category'] = '여성/가족'
    df.loc[240, 'category'] = '여성/가족'
    df.loc[242, 'category'] = '여성/가족'
    df.loc[246, 'category'] = '여성/가족'
    df.loc[250, 'category'] = '여성/가족'
    df.loc[253, 'category'] = '여성/가족'
    df.loc[256, 'category'] = '여성/가족'
    df.loc[259, 'category'] = '여성/가족'
    df.loc[262, 'category'] = '여성/가족'
    df.loc[265, 'category'] = '여성/가족'
    df.loc[268, 'category'] = '여성/가족'
    df.loc[277, 'category'] = '여성/가족'
    df.loc[280, 'category'] = '여성/가족'
    df.loc[356, 'category'] = '여성/가족'
    df.loc[387, 'category'] = '여성/가족'
    df.loc[400, 'category'] = '여성/가족'
    #남성
    df.loc[229, 'category'] = '남성'
    df.loc[231, 'category'] = '남성'
    df.loc[233, 'category'] = '남성'
    df.loc[234, 'category'] = '남성'
    df.loc[235, 'category'] = '남성'
    df.loc[236, 'category'] = '남성'
    df.loc[247, 'category'] = '남성'
    df.loc[249, 'category'] = '남성'
    df.loc[252, 'category'] = '남성'
    df.loc[255, 'category'] = '남성'
    df.loc[258, 'category'] = '남성'
    df.loc[261, 'category'] = '남성'
    df.loc[264, 'category'] = '남성'
    df.loc[267, 'category'] = '남성'
    df.loc[276, 'category'] = '남성'
    df.loc[279, 'category'] = '남성'
    df.loc[346, 'category'] = '남성'
    #성소수자
    df.loc[488, 'category'] = '성소수자'
    df.loc[489, 'category'] = '성소수자'
    df.loc[492, 'category'] = '성소수자'
    df.loc[493, 'category'] = '성소수자'
    df.loc[500, 'category'] = '성소수자'
    df.loc[502, 'category'] = '성소수자'
    df.loc[503, 'category'] = '성소수자'
    df.loc[505, 'category'] = '성소수자'
    df.loc[506, 'category'] = '성소수자'
    df.loc[507, 'category'] = '성소수자'
    df.loc[508, 'category'] = '성소수자'
    #인종
    df.loc[108, 'category'] = '인종/국적'
    df.loc[109, 'category'] = '인종/국적'
    df.loc[110, 'category'] = '인종/국적'
    df.loc[112, 'category'] = '인종/국적'
    df.loc[114, 'category'] = '인종/국적'
    # #국적
    # for i in range(1692,1712):
    #     df.loc[i, 'category'] = '국적'
    #연령
    df.loc[223, 'category'] = '연령'
    df.loc[224, 'category'] = '연령'
    df.loc[225, 'category'] = '연령'
    df.loc[226, 'category'] = '연령'
    df.loc[235, 'category'] = '연령'
    df.loc[248, 'category'] = '연령'
    df.loc[249, 'category'] = '연령'
    df.loc[250, 'category'] = '연령'
    #지역
    df.loc[827, 'category'] = '지역'
    df.loc[828, 'category'] = '지역'
    df.loc[829, 'category'] = '지역'
    df.loc[830, 'category'] = '지역'
    df.loc[831, 'category'] = '지역'
    df.loc[805, 'category'] = '지역'
    df.loc[806, 'category'] = '지역'
    df.loc[808, 'category'] = '지역'
    df.loc[809, 'category'] = '지역'
    #종교
    for i in range(1410,1422):
        df.loc[i, 'category'] = '종교'
    #기타혐오
    df.loc[1379, 'category'] = '기타 혐오'
    df.loc[1380, 'category'] = '기타 혐오'
    df.loc[1385, 'category'] = '기타 혐오'
    df.loc[1377, 'category'] = '기타 혐오'
    #악플
    df.loc[1193, 'category'] = '악플/욕설'
    df.loc[1191, 'category'] = '악플/욕설'
    #욕설
    df.loc[102, 'category'] = '악플/욕설'
    df.loc[100, 'category'] = '악플/욕설'
    #개인지칭
    df.loc[182, 'category'] = '개인지칭'
    df.loc[183, 'category'] = '개인지칭'
    unness = df[df["category"] == 0].index
    df.drop(unness)

def main():
    url = 'https://unicode.org/emoji/charts/full-emoji-list.html'

    html = pd.read_html(url)
    df = html[0]

    col = []
    for i in range(15):
        col.append(df.columns[i][2])

    df.columns = col
    df['Code'] = df['Code'].map(lambda x: x[:7]).copy()
    df['U_Code'] = df['Code'].apply(NotUnitcode)
    emoji = df[~df['U_Code'].isna()][['Code','Browser','CLDR Short Name']].reset_index(drop=True)
    emoji.to_csv('emoji.csv',index = False)
    df = pd.read_csv("emoji.csv")
    
    category(df)
    df.to_csv("emoji_category.csv",index=False)
    
if __name__ == "__main__":
    main()