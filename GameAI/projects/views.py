from django.http import HttpResponse
from django.shortcuts import render
from .models import Paragraphs, Answer


from .OpenAI.GptAPI import Translate
from .MLModel.evaluate import useModel



def main(request):
    return render(request, 'main/main.html')

def result(request):
    if request.method == 'POST':
        paragraph = request.POST.get('paragraph')
        translate = request.POST.get('checkBox')        

        generatedParagraphs = Paragraphs.objects.create(
            detail = paragraph
        )

        answer = 0.0

        context ={}

        if translate:
            translated = Translate(paragraph)
            answer = useModel(translated)
            
            context ={
            "answer": answer,
            "paragraph": paragraph,
            "translated": translated
            }
        else:
            answer = useModel(paragraph)

            context ={
            "answer": answer,
            "paragraph": paragraph
            }
            

        Answer.objects.create(
            answer=answer,
            details=generatedParagraphs
        )           
    
    return render(request, 'main/result.html', context)