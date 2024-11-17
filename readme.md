Rep Structure:
<pre>
|-model_training.ipynb  This is the notebook used to train the custom model
|-audioprocessing  The django project that has the UI.
    |- audio
        |- model-2.pt     The model weights
        |- views.py     The model gets called over here.
    |- templates/audio
        |- record.html  The UI.
</pre>
