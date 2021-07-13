filename = "datalar.xlsx";

data = readtable(filename,'TextType','string');
head(data);
data.film_name = categorical(data.film_name);

figure
histogram(data.film_name);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

cvp = cvpartition(data.film_name,'Holdout',0.2);
dataTrain = data(training(cvp),:);
dataValidation = data(test(cvp),:);

customStopWords = (["," "." ";" ":" ":)" "↵" "(" ")" "!" "'" "^" "#" "%" "/" "{" "}" "[" "]" "?" "\" "$" "-" "_" "|" "acaba" "ama" "ancak" "artık" "asla" "aslında" "az" "bana" "bazen" "bazı" "bazıları" "bazısı" "belki" "ben" "beni" "benim" "beş" "bile" "bir" "birçoğu" "birçok" "birçokları" "biri" "birisi" "birkaç" "birkaçı" "bir" "şey" "bir" "şeyi" "biz" "bize" "bizi" "bizim" "böyle" "böylece" "bu" "buna" "bunda" "bundan" "bunu" "bunun" "burada" "bütün" "çoğu" "çoğuna" "çoğunu" "çok" "çünkü" "da" "daha" "de" "değil" "demek" "diğer" "diğeri" "diğerleri" "diye" "dolayı" "elbette" "enfakat" "falan" "felan" "filan" "gene" "gibi" "hangi" "hangisi" "hani" "hatta" "hem" "henüz" "hep" "hepsi" "hepsine" "hepsini" "her" "her" "biri" "herkes" "herkese" "herkesi" "hiç" "hiç" "kimse" "hiçbiri" "hiçbirine" "hiçbirini" "için" "içinde" "ile" "ise" "işte" "kaç" "kadar" "kendi" "kendine" "kendini" "ki" "kim" "kime" "kimi" "kimin" "kimisi" "madem" "mı" "mi" "mu" "mü" "nasıl" "ne" "ne" "kadar" "ne" "zaman" "neden" "nedir" "nerde" "nerede" "nereden" "nereye" "nesi" "neyse" "niçin" "niye" "ona" "ondan" "onlar" "onlara" "onlardan" "onların" "onu" "onun" "orada" "oysa" "oysaki" "öbürü" "ön" "önce" "ötürü" "öyle" "sana" "sen" "senden" "seni" "senin" "siz" "sizden" "size" "sizi" "sizin" "son" "sonra" "şayet" "şey" "şimdi" "şöyle" "şu" "şuna" "şunda" "şundan" "şunlar" "şunu" "şunun" "tabi" "tamam" "tüm" "tümü" "üzere" "var" "ve" "veya" "veyahutya" "ya" "da" "yani" "yerine" "yine" "yoksa" "zaten" "zira"]);

textDataTrain = dataTrain.comment;
textDataValidation = dataValidation.comment;
YTrain = dataTrain.film_name;
YValidation = dataValidation.film_name;

textDataTrain = lower(textDataTrain);
textDataValidation = lower(textDataValidation);%küçük harf yaptık

str = strrep( textDataTrain, 'ç', 'c' );
str = strrep( str, 'ğ', 'g' );
str = strrep( str, 'ı', 'i' );
str = strrep( str, 'ö', 'o' );
str = strrep( str, 'ş', 's' );
str = strrep( str, 'ü', 'u' );
documents = tokenizedDocument(str);
textDataTrain = removeWords(documents,customStopWords);

str = strrep( textDataValidation, 'ç', 'c' );
str = strrep( str, 'ğ', 'g' );
str = strrep( str, 'ı', 'i' );
str = strrep( str, 'ö', 'o' );
str = strrep( str, 'ş', 's' );
str = strrep( str, 'ü', 'u' );
documents = tokenizedDocument(str);
textDataValidation = removeWords(documents,customStopWords);

figure
wordcloud(textDataTrain);
title("Training Data")

documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);

documentsTrain(1:5)

enc = wordEncoding(documentsTrain); % içindeki kelimelerden bir kelime kodlaması oluşturur documents.

documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")

sequenceLength = 40;
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);
XTrain(1:5);


XValidation = doc2sequence(enc,documentsValidation,'Length',sequenceLength);

inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 80; %Gizli katman sayısı 

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));
bag = bagOfNgrams(documents,'NGramLengths',3);
figure
wordcloud(bag);
title("Text Data: Trigrams")

layers = [ ...
    sequenceInputLayer(inputSize) %Bir dizi giriş katmanı oluşturuyor
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

options = trainingOptions('adam', ...
    'MaxEpochs',3, ...
    'MiniBatchSize',32, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false); %çıktı mesajlarının ayrıntı düzeyini kontrol eder.

net = trainNetwork(XTrain,YTrain,layers,options);
DUYGUanaliz = net;


modelperformans = classify(net,XValidation);
Model = modelStatistics(YValidation, modelperformans);


save DUYGUanaliz






