# Asyèt

Estimer les calories d'un repas à partir d'une photo — y compris les plats haïtiens — **directement sur votre téléphone**, sans compte, sans abonnement et **sans envoyer la photo** sur Internet.

## Comment ça marche
1. Prenez votre assiette en photo : l'appli propose les plats qu'elle reconnaît (diri ak pwa, griyo, bannann peze, pikliz, legim, soup joumou…).
2. Touchez les plats présents, ajustez les grammes : calories, protéines, glucides et lipides se calculent.
3. Un plat manque ? Cherchez-le dans la liste (85 plats haïtiens et internationaux).
4. Profil (IMC, besoins du jour), journal du jour et objectif — gardés dans le téléphone.

Estimation à partir d'une photo, pas une mesure exacte. Pour un régime médical, demandez l'avis d'un professionnel de santé.

## Ajouter un plat
Ajoutez une ligne dans `plats.json` (nom FR/HT, description en anglais pour la reconnaissance, valeurs pour 100 g), puis recalculez `plats-emb.json` (signatures CLIP du texte).

## Modèle et licences
- Reconnaissance : CLIP ViT-B/32 (OpenAI) — licence MIT ; encodeur d'image ONNX 4 bits (`clip-vision-q4.onnx`, conversion Xenova).
- Valeurs nutritionnelles : moyennes indicatives (tables de composition usuelles) pour des portions courantes.
- Exécution : ONNX Runtime Web (MIT), coi-serviceworker (MIT).
