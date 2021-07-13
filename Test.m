a=load("DUYGUanaliz.mat")
reportsNew = [ ...
    "berbat ötesi bir film hayatımda böyle film görmedim gerçekten çok kötü bu kadar iğrenç olmasını beklemiyordum"
    "güzel bir film ama sıkıcı olduğu noktalar çoktu kararsız kaldım."
    "harika ötesi muhteşem film herkes kesinlikle izlesin tavsiye ederims."];
documentsNew = preprocessText(reportsNew);
XNew = doc2sequence(a.enc,documentsNew,'Length',a.sequenceLength);
labelsNew = classify(a.net,XNew)