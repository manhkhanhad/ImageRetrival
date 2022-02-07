<h1 align="center">Welcome to ImageRetrival 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://twitter.com/manhkhanhad" target="_blank">
    <img alt="Twitter: manhkhanhad" src="https://img.shields.io/twitter/follow/manhkhanhad.svg?style=social" />
  </a>
</p>

> This project provides a flask application for image retrieval. It consists of 3 modules: 
>
>* Deep Image retrieval, which allows users to perform visual searches both on the query images and any new image. 
>
>* Spatial Reranking for reranking the results based on local feature matching and finding query regions on the target image.
>
>* Locality-sensitive hashing (LSHash) for coping with the large-scale dataset.

# ✨ Demo

<p align="center">
  <img width="700" align="center" src="demo/demo.gif" alt="demo"/>
</p>

# 🔧 Install 
## 💻 Front-end
Make sure you have Nodejs, Npm and Yarn installed \
Download the code and install package
```
// Download sourcecode
git clone https://github.com/vinhqngo5/CS336-ImageRetrieval.git

// Install required package with Yarn
yarn install
```
To start frontend run
```
yarn start
```

## 💻 Back-end
Use conda to create enviroment from `environment.yml` file
```
conda env create -f environment.yml
conda activate myenv
```
Download the code
```
git clone https://github.com/manhkhanhad/ImageRetrival.git
```
Setup Image Retrieval module, please follow [these instructions](https://github.com/naver/deep-image-retrieval/blob/master/README.md)


Setup Image Spatial Re-ranking module, please follow [these instructions](https://github.com/ducha-aiki/affnet/blob/master/README.md)

Start backend
```
sh start_backend.sh
```


# 🚀 Usage

Go to http://localhost:3000 in your browser and experience.

# 👤 Author

**Manh-Khanh Ngo Huu** [manhkhanhad](https://github.com/manhkhanhad)

**Huy Nguyen** 
[akaRainbowShine](https://github.com/akaRainbowShine)

**Vinh Quang Ngo**
[vinhqngo5](https://github.com/vinhqngo5)

# 🔰 References
Special thanks to Jerome Revaud, Rafael de Rezende, Cesar de Souza, Diane Larlus, and Jon Almazan at [Naver Labs Europe](https://europe.naverlabs.com/). In this library, we have used the code and pretrained model for Retrieval module from their awesome [Image Retrieval](https://github.com/naver/deep-image-retrieval) repository.

Special thanks to Dmytro Mishkin, Filip Radenovic, Jiri Matas. We have used their code ([Image Matching](https://github.com/ducha-aiki/affnet)) to make the Spatial Reranking module.

Special thanks to [Loreto Parisi](https://github.com/loretoparisi) for his awesome [LSHash library](https://github.com/loretoparisi/lshash)

# Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_