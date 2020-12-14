#!/usr/bin/env python
# coding: utf-8

import numpy as np

# import sys
# sys.path.append("/path/to/aux_classifier")
import aux_classifier.extraction as extraction
import aux_classifier.data_loader as data_loader
import aux_classifier.utils as utils

import matplotlib.pyplot as plt
import argparse


def plot_neurons_per_layer(toplayers, title, numberlayers=13, layersize = 768):
    nlayers = np.floor_divide(toplayers,layersize)
    (unique, counts) = np.unique(nlayers, return_counts=True)
    layersLabels = np.arange(numberlayers)
    layersCounts = np.zeros(numberlayers)
    for i in range(len(unique)):
        layersCounts[unique[i]]=counts[i]
    plt.bar(layersLabels, layersCounts, align='center', alpha=0.5)
    plt.xticks(layersLabels)
    plt.ylabel('Counts')
    plt.title(title)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='qarib/bert-base-qarib', help="Name of model")
    parser.add_argument("--input_corpus", help="Text file path with one sentence per line")
    parser.add_argument("--test_corpus", help="Text file path with one sentence per line")

    args = parser.parse_args()

    extraction.extract_representations(args.model_name,
                                       args.input_corpus,
                                       args.input_corpus+'.hdf5',
                                       aggregation="average" , output_type="hdf5"
                                      )

    # extraction.extract_representations(args.model_name,
    #                                    args.input_corpus,
    #                                    args.input_corpus+'.json',
    #                                    aggregation="average" #, output_type="hdf5"
    #                                   )
    # Loading QARiB
    activations_path=args.input_corpus+'.hdf5'
    activations, num_layers = data_loader.load_activations(activations_path, 768, 512)
    print("hdf5 len:",len(activations))

    # activations_path=args.input_corpus+'.json'
    # activations, num_layers = data_loader.load_activations(activations_path, 768, 512)
    # print("json len:",len(activations))

    tokens = data_loader.load_data(args.input_corpus,
                                   args.input_corpus+'.label',
                                   activations,
                                   512
                                  )


    X, y, mapping = utils.create_tensors(tokens, activations, 'LABEL1')
    label2idx, idx2label, src2idx, idx2src = mapping



    model = utils.train_logreg_model(X, y, lambda_l1=0.001, lambda_l2=0.001)


    # Load test data
    extraction.extract_representations(args.model_name,
                                       args.test_corpus,
                                       args.test_corpus+'.hdf5',
                                       aggregation="average" #, output_type="hdf5"
                                      )
    activations_test, num_layers = data_loader.load_activations(args.test_corpus+'.hdf5', 768, 512)
    tokens_test = data_loader.load_data(args.input_corpus,
                                   args.test_corpus+'.label',
                                   activations_test,
                                   512
                                  )

    X_test, y_test, _ = utils.create_tensors(tokens_test, activations_test, 'LABEL1')

    res = utils.evaluate_model(model, X_test, y_test, idx_to_class=idx2label)


    print(res['__OVERALL__'])


    ordering, cutoffs = utils.get_neuron_ordering(model, label2idx)


    nlen = len(ordering)


    #plot_neurons_per_layer(ordering[:cutoffs[5]],title='Labels per layer in QARiB POS')



    for i in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75 ]:
        X_selected = utils.filter_activations_keep_neurons(ordering[:int(nlen*i)], X)

        print(X_selected.shape)
        model_selected = utils.train_logreg_model(X_selected, y, lambda_l1=0.001, lambda_l2=0.001)

        eval_selected = utils.evaluate_model(model_selected, X_test, y_test, idx_to_class=idx2label)

        print(i, eval_selected['__OVERALL__'], res['__OVERALL__'], (res['__OVERALL__']-eval_selected['__OVERALL__']))



if __name__ == '__main__':
    main()


# # # Further analysis

# # In[ ]:


# top_neurons = utils.get_top_neurons(model, 0.01, label2idx)


# # In[ ]:


# top_neurons


# # In[ ]:


# top_neurons[1].keys()


# # In[ ]:


# for tag in {'NOUN', 'V', 'PART', 'PREP', 'ADJ', 'PREP+DET+NOUN', 'V+PRON', 'DET+ADJ+NSUFF' , 'PART+PRON' }:
#     plot_neurons_per_layer(top_neurons[1][tag],title='Labels per layer in QARiB '+tag)


# # In[ ]:


# plot_neurons_per_layer(top_neurons[1]['DET+NOUN'],title='Labels per layer in QARiB DET+NOUN')


# # In[ ]:


# X_zeroed = utils.zero_out_activations_keep_neurons(ordering[:10], X)
# utils.evaluate_model(model, X_zeroed, y, idx_to_class=idx2label)


# # In[ ]:


# activations


# # In[ ]:


# import aux_classifier.visualization as visualization
# visualization.visualize_activations("في إجراءات استثنائية بسبب وباء كورونا .", activations[0][:, ordering[0]])


# # In[ ]:


# import svgwrite
# FONT_SIZE = 20
# MARGIN = 10
# CHAR_LIMIT = 61
# def break_lines(text, limit=50):
#     lines = []
#     curr_line = ""
#     for token in text.split(' '):
#         if len(curr_line) + 1 + len(token) < limit:
#             curr_line += token + " "
#         else:
#             lines.append(curr_line[:-1])
#             curr_line = token + " "
#     lines.append(curr_line[:-1])
#     return lines

# def get_rect_style(color, opacity):
#     return """opacity:%0.5f;
#             fill:%s;
#             fill-opacity:1;
#             stroke:none;
#             stroke-width:0.26499999;
#             stroke-linecap:round;
#             stroke-linejoin:miter;
#             stroke-miterlimit:4;
#             stroke-dasharray:none;
#             stroke-dashoffset:0;
#             stroke-opacity:1""" % (opacity, color)

# def get_text_style(font_size):
#     return """font-style:normal;
#             font-variant:normal;
#             font-weight:normal;
#             font-stretch:normal;
#             font-size:%0.2fpx;
#             line-height:125%%;
#             font-family:monospace;
#             -inkscape-font-specification:'Arial Unicode MS, Normal';
#             font-variant-ligatures:none;
#             font-variant-caps:normal;
#             font-variant-numeric:normal;
#             text-align:start;
#             writing-mode:lr-tb;
#             text-anchor:start;
#             stroke-width:0.26458332px""" % (font_size)


# # In[ ]:


# text = 'في إجراءات استثنائية بسبب وباء كورونا .'
# lines = break_lines(text, limit=CHAR_LIMIT)
# char_width = FONT_SIZE*0.59
# char_height = FONT_SIZE*1.25


# # In[ ]:


# lines


# # In[ ]:


# image_height = len(lines) * char_height * 1.2
# image_width = CHAR_LIMIT * char_width

# dwg = svgwrite.Drawing("tmp.svg", size=(image_width, image_height),
#                     profile='full')
# dwg.viewbox(0, 0, image_width, image_height)


# # In[ ]:


# darken=2
# colors=["#d35f5f", "#00aad4"]
# scores = activations[0][:, ordering[0]]
# offset = 0


# # In[ ]:


# group = dwg.g()
# for _ in range(darken):
#     word_idx = 0
#     for line_idx, line in enumerate(lines):
#         char_idx = 0
#         max_score = max(scores)
#         min_score = abs(min(scores))
#         limit = max(max_score, min_score)
#         for word in line.split(' '):
#             print('W:',word)
#             score = scores[word_idx]
#             if score > 0:
#                 color = colors[1]
#                 opacity = score/limit
#             else:
#                 color = colors[0]
#                 opacity = abs(score)/limit

#             for _ in word:
#                 rect_insert = (0 + char_idx * char_width, offset + 7 + line_idx * char_height)
#                 rect_size = ("%.2fpx"%(char_width), "%0.2fpx"%(char_height))
#                 group.add(
#                     dwg.rect(insert=rect_insert,
#                             size=rect_size,
#                             style=get_rect_style(color, opacity)
#                             )
#                 )
#                 char_idx += 1

#             final_rect_insert = (0 + char_idx * char_width, offset + 7 + line_idx * char_height)
#             final_rect_size = ("%.2fpx"%(char_width), "%0.2fpx"%(char_height))
#             group.add(
#                 dwg.rect(insert=final_rect_insert,
#                         size=final_rect_size,
#                         style=get_rect_style(color, opacity)
#                         )
#             )

#             char_idx += 1
#             word_idx += 1

#     for line_idx, line in enumerate(lines):
#         text_insert = (0, offset + FONT_SIZE*1.25*(line_idx+1))
#         print(text_insert,text)
#         text = dwg.text(text,
#                         insert=text_insert,
#                         fill='black',
#                         style=get_text_style(FONT_SIZE))
#         group.add(text)
# offset += FONT_SIZE*1.25*len(lines) + MARGIN

# dwg.add(group)


# # In[ ]:


# dwg


# In[ ]:




