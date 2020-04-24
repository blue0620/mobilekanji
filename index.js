const webcamElement = document.getElementById('webcam');
let net;
async function app() {
  console.log('Loading mobilenet..');
  // Load the model.
  net = await tf.loadLayersModel('./model.json');//モデルの読み込み
  console.log('Successfully loaded model');
	//ラベルファイルの読み込み
	const CLASSES =await fetch("label.json").then(response =>response.json());
	const WEBCAM_CONFIG = {"facingMode": "environment"};//背面のカメラを使う
	const webcam = await tf.data.webcam(webcamElement,WEBCAM_CONFIG);
	while (true) {
	    const imgEl = await webcam.capture();//ウェブカメラからキャプチャする
			const example = imgEl.reshape([-1, 64, 64, 3]).div(255.0); 
		  const prediction = await net.predict(example).dataSync();
		  let results = Array.from(prediction)
		                .map(function(p,i){
		    return {
		        className: String.fromCharCode(CLASSES[i].replace('U', '0x')),//unicodeを漢字に変換
						probability: p.toFixed(2)//小数点2桁まで
		    };
		    }).sort(function(a,b){//probabirityが大きい順にソート
		        return b.probability-a.probability;
		    }).slice(0,3);//大きい方から3つ取る
			document.getElementById("resultview").textContent = JSON.stringify(results, null, 4);
	    imgEl.dispose();
	    await tf.nextFrame();
  }
}

app();