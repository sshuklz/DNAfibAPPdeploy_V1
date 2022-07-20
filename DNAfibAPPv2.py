import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import (Input, Output, State)
import dash_bootstrap_components as dbc

import base64
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import re
from skimage import io

from ImageOPs import (ImageOperations, parse_contents, blank_fig)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
server = app.server

app.config['suppress_callback_exceptions'] = True
app.title = 'DNA Fiber Analysis DEMO'

DNA_fiber_types = ['stalled',
                '2nd origin',
                'progressing fork one direction',
                'progressing fork bidirectional',
                'terminated fork',
                'interspersed']

color_types = ['Primary Red : Secondary Green',
               'Primary Green : Secondary Red',
               'Primary Red : Secondary Blue',
               'Primary Blue : Secondary Red',
               'Primary Green : Secondary Blue',
               'Primary Blue : Secondary Green',
               
              'Primary Yellow : Secondary Blue',
              'Primary Blue : Secondary Yellow',
              'Primary Magenta : Secondary Green',
              'Primary Green : Secondary Magenta',
              'Primary Red : Secondary Cyan',
              'Primary Cyan : Secondary Red',
              
              'Primary Yellow : Secondary Magenta',
              'Primary Magenta : Secondary Yellow',
              'Primary Cyan : Secondary Magenta',
              'Primary Magenta : Secondary Cyan',
              'Primary Cyan : Secondary Yellow',
              'Primary Yellow : Secondary Cyan']

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'fontWeight': 'bold',
    'padding': '6px'
}

df = pd.DataFrame(columns=['Selection', 'Type', 'Height', 'Width', 'Green:Red'])

columns=['Selection', 'Type', 'Height', 'Width', 'Green:Red']

colors = [
 ['R','G'],
 ['G','R'],
 ['R','B'],
 ['B','R'],
 ['G','B'],
 ['B','G'],
  ['Y','B'],
  ['B','Y'],
  ['M','G'],
  ['G','M'],
  ['R','C'],
  ['C','R'],
   ['Y','M'],
   ['M','Y'],
   ['C','M'],
   ['M','C'],
   ['C','Y'],
   ['Y','C']]

color_options = []

height_vals = [10,10,30,10,10,30]

for i in range(len(colors)):

    c1 = colors[i][0] ; c2 = colors[i][1]
    label_name = c1 +c2 + '.png'

    color_options.append(
    {
        "label": html.Div(
            [
                html.Img(src="/assets/Color_options/Label_" + label_name, height=30),
                html.Div(color_types[i], style={'font-size': 15, 'padding-left': 10}),
            ], style={'display': 'flex', 'align-items': 'left', 'justify-content': 'left'}
        ),
        "value": color_types[i],
    })

def fiber_dropdown_images(c1,c2):

    fiber_options = []

    for i in range(6):
    
        label_name = DNA_fiber_types[i].replace(" ", "_") + '_' + c1 + '_' + c2 + '.png'
    
        fiber_options.append(
            {
                "label": html.Div(
                    [
                        html.Img(src="/assets/DNA_Fib_type_colors/" + label_name, height=height_vals[i]),
                        html.Div(DNA_fiber_types[i], style={'font-size': 15, 'padding-left': 10}),
                    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between',}
                ),
                "value": DNA_fiber_types[i],
            })
        
    return(fiber_options)

items_method = [
    dbc.DropdownMenuItem("Rectangle"),
    dbc.DropdownMenuItem("Lasso"),
    dbc.DropdownMenuItem("Line"),
]

fig = px.imshow(io.imread("assets/blank.png"))

fig.update_layout(
    coloraxis_showscale=False, 
    width=1000, height=750, 
    margin=dict(l=0, r=0, b=0, t=0)
)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(dragmode=False)

app.layout = html.Div([
    
    html.Meta(charSet='UTF-8'),
    html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),
    
    dcc.Store(id='shape_coords', storage_type='memory'), 
    dcc.Store(id='shape_number', storage_type='memory'), 

        html.Div([
            
            html.Div(
                id='title-app', 
                children=[html.H1(app.title)],
                style={'textAlign' : 'center', 'paddingTop' : 0}),
            
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                    style={
                        'width': '100%',
                        'height': '70px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                        'backgroundColor': '#F0F1F1'
                    },
                    multiple=True),
                ], style={'paddingTop' : 30}),
            
            html.Div([
                
                dcc.Tabs(
                    id='image-processors-tabs',
                    value='operators',
                    children=[
                        
                        dcc.Tab(
                            label='Image Operations',
                            value='operators',
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[
                                
                                html.H6('Gamma'),
                                dcc.Slider(
                                    id='slider-Gamma',
                                    min=0.01,
                                    max=2,
                                    step=0.01,
                                    value=1,
                                    marks={i: str(i) for i in range(0, 2, 1)}),

                                html.H6('Correct Red Channel'),
                                dcc.Slider(
                                    id='slider-CR',
                                    min=0,
                                    max=256,
                                    step=1,
                                    value=0,
                                    marks={i: str(i) for i in range(0, 256, 16)}),
                                
                                html.H6('Correct Green Channel'),
                                dcc.Slider(
                                    id='slider-GR',
                                    min=0,
                                    max=256,
                                    step=1,
                                    value=0,
                                    marks={i: str(i) for i in range(0, 256, 16)}),
                                           
                                html.H6('Correct Blue Channel'),
                                dcc.Slider(
                                    id='slider-BR',
                                    min=0,
                                    max=256,
                                    step=1,
                                    value=0,
                                    marks={i: str(i) for i in range(0, 256, 16)}),
                                           
                                html.H6('Denoise'),
                                dcc.Slider(
                                    id='slider-DI',
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=0,
                                    marks={i: str(i) for i in range(0, 50, 10)}),
                                        
                                html.H6('Contrast'),
                                dcc.Slider(
                                    id='slider-Contrast',
                                    min=1,
                                    max=3,
                                    step=0.1,
                                    value=1,
                                    marks={i: str(i) for i in range(1, 3, 1)}),
                                     
                            ]),
                    
                        dcc.Tab(
                            label='Label Colors and Fiber Details',
                            value='color_tab',
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[html.Div([
                                    
                                html.H6('Label Colors'),
                                html.Div([dcc.Dropdown(color_options, 
                                                       color_types[0],
                                                       id='color_label-dropdown')])
                            
                            ])]
                            
                        ),
                    
                        dcc.Tab(
                            label='Image Selections',
                            value='selections',
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[html.Div([
                                    
                                html.H6('Selection type'),
                                html.Div(html.Div([
                                    
                                    dcc.Dropdown(['Rectangle', 'Lasso', 'Line'], 
                                                 'Rectangle', id='method-dropdown')
                                    
                                ], style={'width':'100%'})),
                                          
                                html.H6('DNA Fiber type'),
                                html.Div([
                                    
                                    dcc.Dropdown(fiber_dropdown_images('R','G'),
                                                       DNA_fiber_types[0], 
                                                       id='fiber-dropdown')
                                
                                ], style={'width':'100%','paddingBottom' : 25}),
                                    
                                dash_table.DataTable(
                                    id="annotations-table",
                                    columns=[
                                        
                                        dict(name=n,
                                            id=n,
                                            presentation=("input"))
                                        for n in columns
                                    
                                    ],
                                    editable=True,
                                    page_action="native",
                                    page_current= 0,
                                    page_size= 10,
                                    style_data={"height": 15},
                                    style_cell={
                                        "textAlign": "left",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "maxWidth": 0,
                                    },
                                    fill_width=True,
                                ),
                                
                            ])]
                        )
                        
                    ]
                    
                )
                
            ], className='tab-div')
            
        ], className='flex-item-left'),

    html.Div(html.Div([html.Div( 
            
        children=[
            
            html.H4('Image Used - Output'),
            dcc.Loading(
                id='loading-op',
                type='dot',
                children=html.Div(html.Div([html.Div([
                    
                    dcc.Graph(id='out-op-img', 
                              figure=fig,
                              config={'modeBarButtonsToAdd':['eraseshape'],
                                      'modeBarButtonsToRemove':['zoom2d',
                                                                'pan2d',
                                                                'zoomIn2d',
                                                                'zoomOut2d',
                                                                'autoScale2d',
                                                                'resetScale2d'],
                                      'displaylogo':False})
                    
                ], style={'paddingTop' : 50,})]))
            )
            
        ],style={'textAlign' : 'center', 'paddingTop' : 50}
            
    )]), className='flex-item-right'),

    html.Div(html.Div([html.Div( 
        
        children= [
            html.H4(' ', id='Selected_Fiber_Title'),
            dcc.Loading(
                id='loading-sel',
                type='dot',
                children=html.Div((html.Div([html.Div([
                    
                    dcc.Graph(id='sel-op-img', 
                              figure=blank_fig(),
                              config={'displayModeBar' : False})
                    
                ], style={'paddingTop' : 12,})])))
            )
            
        ], style={'textAlign' : 'center', 'paddingTop' : 50}
    
    )]), className='flex-item-right')

], className='flex-container')



@app.callback(
    Output('out-op-img', 'relayoutData'),
    Input('image-processors-tabs', 'value'))

def reset_relayout(tab):
    
    if tab == 'selections':
        
        return dash.no_update
    
    return None



@app.callback(
    Output('Selected_Fiber_Title', 'children'),
    Input('image-processors-tabs', 'value'))

def show_text_selection_title(tab):
    
    if tab == 'selections':
        
        return "Selected Fiber"
    
    return " "



@app.callback(
    Output("fiber-dropdown", "options"),
    Input("color_label-dropdown", "value"))

def color_fiber_display(color_selection):
    
    if color_selection is None:
        
        return dash.no_update
    
    i = color_types.index(color_selection)
    c1 = colors[i][0] ; c2 = colors[i][1]
    
    return fiber_dropdown_images(c1,c2)



@app.callback(
    Output('out-op-img', 'figure'),
    [Input('upload-image', 'contents'),
     Input("slider-Gamma", "value"),
     Input("slider-CR", "value"),
     Input("slider-GR", "value"),
     Input("slider-BR", "value"),
     Input("slider-DI", "value"),
     Input("slider-Contrast", "value"),
     Input('image-processors-tabs', 'value'),
     Input('method-dropdown', 'value'),
     State('out-op-img', 'figure'),
     State('upload-image', 'filename')])

def get_operated_image(contents, gam, CR, GR, BR, DI, con, tab, method, filenames, dates):
    
    if contents is not None:
        
        imsrc = parse_contents(contents, filenames, dates)
        imo = ImageOperations(image_file_src = imsrc)
        out_img = imo.read_operation()
        
        if gam != 1:
        
            out_img = imo.gamma_operation(thresh_val = gam)
            
        if CR > 0:
        
            out_img = imo.CR_operation(thresh_val = CR)
            
        if GR > 0:
        
            out_img = imo.GR_operation(thresh_val = GR)
            
        if BR > 0:
        
            out_img = imo.BR_operation(thresh_val = BR)
            
        if DI > 0:
        
            out_img = imo.denoiseI_operation(thresh_val = DI)
            
        if con > 1:
        
            out_img = imo.contrast_operation(thresh_val = con)

        out_image_fig = px.imshow(out_img)
        out_image_fig.update_layout(
            coloraxis_showscale=False, 
            width=1000, height=750, 
            margin=dict(l=0, r=0, b=0, t=0)
        )
        out_image_fig.update_xaxes(showticklabels=False)
        out_image_fig.update_yaxes(showticklabels=False)
        out_image_fig.update_layout(dragmode=False)
        
        if tab == 'selections':
            
            if method == 'Rectangle':
                
                out_image_fig.update_layout(dragmode='drawrect',
            newshape=dict(line = {"color": "#0066ff", "width": 1.5, "dash": "solid"}))
                
            if method == 'Lasso':
                
                out_image_fig.update_layout(dragmode='drawclosedpath',
            newshape=dict(line = {"color": "#0066ff", "width": 1.5, "dash": "solid"}))
                
            if method == 'Line':
                
                out_image_fig.update_layout(dragmode='drawline',
            newshape=dict(line = {"color": "#0066ff", "width": 1.5, "dash": "solid"}))
                    
        return out_image_fig
    
    else:
        
        return dash.no_update



@app.callback(
    [Output("annotations-table", "data"),
     Output('shape_number', 'data'), 
     Output('shape_coords', 'data')], 
    [Input('out-op-img', 'relayoutData'),
     Input('out-op-img', 'figure'),
     Input('image-processors-tabs', 'value'),
     Input('fiber-dropdown', 'value'),
     Input('shape_coords', 'data'),
     Input('shape_number', 'data')],
    State("annotations-table", "data"),prevent_initial_call=True)

def shape_added(fig_data, fig, tab, fiber, shape_coords, shape_number, new_row): 
    
    if tab == 'selections':
        
        nparr = np.frombuffer(
            base64.b64decode(fig['data'][0]['source'][22:]), np.uint8)
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imo = ImageOperations(image_file_src=img)
        
        if fig_data is None:
            
            return None, None, None
        
        if fiber is None:
            
            return None, None, None
        
        if 'shapes' in fig_data:
            
            last_shape = fig_data["shapes"][-1]
            x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
            x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
            
            if x0 > x1:
                
                x0, x1 = x1, x0
                
            if y0 > y1:
                
                y0, y1 = y1, y0
            
            Height = int(abs(y1 - y0))
            Width = int(abs(x1 - x0))
            ratio = imo.G_R_operation(x0, x1, y0, y1)
            
            if new_row is None:
            
                n = 0
                new_row = [{'Selection':n, 
                            'Type':fiber, 
                            'Height': Height, 
                            'Width': Width, 
                            'Green:Red':ratio}]
                
            else:
                
                if len(fig_data["shapes"]) < shape_number:
                    
                    shape_coord = []; table_coord = []
                    
                    for shape in fig_data["shapes"]:
                        
                        x0, y0 = int(shape["x0"]), int(shape["y0"])
                        x1, y1 = int(shape["x1"]), int(shape["y1"])
                        
                        if x0 > x1:
                            
                            x0, x1 = x1, x0
                            
                        if y0 > y1:
                            
                            y0, y1 = y1, y0
                        
                        shape_coord += [[x0,x1,y0,y1]]
                    
                    for coord in shape_coords:
                            
                        table_coord += [[coord['x0'],
                                         coord['x1'],
                                         coord['y0'],
                                         coord['y1']]]
                            
                    for i in table_coord:
                        
                        if i not in shape_coord:
                            
                            x0 = i[0] ; x1 = i[1] ; y0 = i[2] ; y1 = i[3]
                            
                    for coord in shape_coords:
                        
                        if [coord['x0'],coord['x1'],coord['y0'],coord['y1']] == [x0,x1,y0,y1]:
                            
                            new_row = list(filter(lambda i: i['Selection'] != coord['n'], new_row))
                            shape_coords = list(filter(lambda i: i['n'] != coord['n'], shape_coords))

                            return new_row, len(fig_data["shapes"]), shape_coords
                        
                if new_row[-1] == {'Selection':new_row[-1]['Selection'],
                                   'Type':fiber, 
                                   'Height': Height, 
                                   'Width': Width, 
                                   'Green:Red':ratio}:
                    
                    return dash.no_update
                
                n = new_row[-1]['Selection'] + 1
                
                new_row.append({'Selection':n, 
                                'Type':fiber, 
                                'Height': Height, 
                                'Width': Width, 
                                'Green:Red':ratio})
            
        elif re.match("shapes\[[0-9]+\].x0", list(fig_data.keys())[0]):
            
            for key, val in fig_data.items():
                
                shape_nb, coord = key.split(".")
                shape_nb = shape_nb.split(".")[0].split("[")[-1].split("]")[0]
                
                if coord == 'x0':
                    x0 = int(fig_data[key])
                    
                elif coord == 'x1':
                    x1 = int(fig_data[key])
                
                elif coord == 'y0':
                    y0 = int(fig_data[key])
                
                elif coord == 'y1':
                    y1 = int(fig_data[key])
            
            if x0 > x1:
                
                x0, x1 = x1, x0
                
            if y0 > y1:
                
                y0, y1 = y1, y0
            
            Height = int(abs(y1 - y0))
            Width = int(abs(x1 - x0))
            ratio = imo.G_R_operation(x0, x1, y0, y1)
            n = int(shape_nb)
            
            new_row[int(shape_nb)]['Selection'] = n
            new_row[int(shape_nb)]['Type'] = fiber
            new_row[int(shape_nb)]['Height'] = Height
            new_row[int(shape_nb)]['Width'] = Width
            new_row[int(shape_nb)]['Green:Red'] = ratio
        
        if shape_coords is None:
            
            shape_coords = [{'n':n, 'x0':x0, 'y0': y0, 'x1': x1, 'y1':y1}]
            
        else:
            
            shape_coords.append({'n':n, 'x0':x0, 'y0': y0, 'x1': x1, 'y1':y1})
        
        return new_row, len(fig_data["shapes"]), shape_coords
    
    else:
        
        return None, None, None

@app.callback(
    Output('sel-op-img', 'figure'), 
    [Input('out-op-img', 'relayoutData'),
     Input('out-op-img', 'figure'),
     Input('image-processors-tabs', 'value'),
     Input('shape_coords', 'data')],prevent_initial_call=True)

def selection_fiber_image(fig_data, fig, tab, shape_coords): 
    
    if tab == 'selections':
        
        if fig_data is None:
            
            return blank_fig()
        
        nparr = np.frombuffer(base64.b64decode(fig['data'][0]['source'][22:]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imo = ImageOperations(image_file_src=img)
        out_img = imo.read_operation()
        
        x0, y0 = shape_coords[-1]['x0'], shape_coords[-1]['y0']
        x1, y1 = shape_coords[-1]['x1'], shape_coords[-1]['y1']
        
        out_img = imo.crop_operation(x0,x1,y0,y1)
        out_image_fig = px.imshow(out_img)
        out_image_fig.update_layout(height=750,
            coloraxis_showscale=False, 
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        out_image_fig.update_xaxes(showticklabels=False)
        out_image_fig.update_yaxes(showticklabels=False)
        out_image_fig.update_layout(dragmode=False)
        
        return out_image_fig
    
    else:
        
        return blank_fig()
        
@app.callback(
    Output("annotations-table", "style_data_conditional"),
    Input('shape_number', 'data'),
    Input('out-op-img', 'hoverData'),
    Input('shape_coords', 'data')
)

def style_selected_rows(shape_number, hover_data, shape_coords):
    
    if shape_number is None:
        
        return dash.no_update
    
    if hover_data is not None:
        
        x0 = hover_data["points"][0]["x"] ; y0 = hover_data["points"][0]["y"]
        
        for i in range(len(shape_coords)):
            
            if x0 >= shape_coords[i]['x0'] and x0 <= shape_coords[i]['x1']:
                
                if y0 >= shape_coords[i]['y0'] and y0 <= shape_coords[i]['y1']:
                    
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{{Selection}} = {}'.format(shape_coords[i]['n']),
                            },
                            'backgroundColor': '#0074D9',
                            'color': 'white'
                        }
                    ]
    
                    return style_data_conditional
    
if __name__ == '__main__':
    app.run_server(debug=True)