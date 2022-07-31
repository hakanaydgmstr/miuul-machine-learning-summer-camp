def outlier_thresholds(dataframe, num_col, q1=0.25, q3=0.75):
    quartile1 = dataframe[num_col].quantile(q1)
    quartile3 = dataframe[num_col].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr

    # outliers = [dataframe[(dataframe[num_col] < low) | (dataframe[num_col] > up)]]
    return low_limit, up_limit


def check_outlier(dataframe, num_col):
    low_limit, up_limit = outlier_thresholds(dataframe, num_col)

    if dataframe[(dataframe[num_col] < low_limit) | (dataframe[num_col] > up_limit)].any(axis=None):
        return True
    else:
        return False


def reach_outliers(dataframe, num_col, index=False):
    low_limit, up_limit = outlier_thresholds(dataframe, num_col)

    if len(dataframe[(dataframe[num_col] < low_limit) | (dataframe[num_col] > up_limit)]) > 10:
        print(dataframe[(dataframe[num_col] < low_limit) | (dataframe[num_col] > up_limit)].head())
    else:
        print(dataframe[(dataframe[num_col] < low_limit) | (dataframe[num_col] > up_limit)])

    if index:
        return dataframe[(dataframe[num_col] < low_limit) | (dataframe[num_col] > up_limit)].index


def remove_outliers(dataframe, num_col):
    low_limit, up_limit = outlier_thresholds(dataframe, num_col)
    df_without_outliers = dataframe[~((dataframe[num_col] < low_limit) | (dataframe[num_col] > up_limit))]

    return df_without_outliers


def replace_with_thresholds(dataframe, num_col):
    low_limit, up_limit = outlier_thresholds(dataframe, num_col)

    dataframe.loc[(dataframe[num_col] < low_limit), num_col] = low_limit
    dataframe.loc[(dataframe[num_col] > up_limit), num_col] = up_limit


def missing_values_table(dataframe, na_name=False):
    # eksik veri bulunduran değişkenlerin seçilmesi
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    # değişkenlerdeki eksik değer miktarı
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    # değişkenlerdeki eksik değer oranı
    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending=False)

    # yukardaki hesapladığımız miktar ve oran bilgilerini içeren bir df oluşturulması
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end='\n')

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()  # dataframe'in kopyasını oluşturduk. Bunun üzerinden işlem yapacağız.

    for col in na_columns:  # daha önceden belirlediğimiz na içeren değişkenlerin bulunduğu listede gez
        # bu değişkenler içerisinde na ifade bulunan satıra 1 diğerlerine 0 yaz, bu değişkenleri de ismine na_flag ifadesini ekleyerek al
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    # temp_df içerisinde '_NA_' stringini içeren değişkenleri al, na_flags'e ata.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns

    for col in na_flags:  # na_flags içerisinde gez
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
        # na içeren değişkenlerin kırılımında bağımlı değişkenin (target'in) ortalamasını ve toplamını içeren bir df oluştur


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    # fonksiyona girilen dataframe içerisinden kategorik değişkenleri one hot encoding işlemine sokuyoruz
    # sonrasında ana dataframe'e bunu kaydederek fonksiyon sonunda bu df'i döndürüyoruz.
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))  # değişkenin içerisinde kaç tane sınıf var?

        # değişkendeki sınıfların toplamları, oranları ve target değişkenine göre oranlarını veren dataframe
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    # dataframe'in kopyasını aldık
    temp_df = dataframe.copy()

    # eğer değişken kategorikse ve içerisindeki sınıflardan herhangi birisinin oranı rare_perc'de verilen orandan az ise
    # bu değişkeni rare_columns'a ata
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O' and
                    (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:  # seçilen rare değişkenler içerisinde gez
        tmp = temp_df[var].value_counts() / len(temp_df)  # temp_df de var değişkeninin oranını al
        rare_labels = tmp[tmp < rare_perc].index  # eğer alınan oran rare_perc'den küçükse bunun label'ını al
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])  # bu label yerine 'Rare' yazdır

    return temp_df


