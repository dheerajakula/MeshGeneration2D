function viewPointSet(point_id) {

    var raw_vertices = document.getElementById(point_id).value;
    console.log(raw_vertices);

    raw_vertices = raw_vertices.split(" ").filter(function (el) {
        return el.length != 0
    });
    console.log(raw_vertices);

    raw_vertices = raw_vertices.map(function (e) {
        e = parseFloat(Number(e));
        return e;
    });
    console.log(raw_vertices);


    var trace = {
        x: raw_vertices.filter((a, i) => i % 2 === 0),
        y: raw_vertices.filter((a, i) => i % 2 === 1),
        mode: 'markers',
        type: 'scatter'
    };
    data.push(trace);
    Plotly.newPlot('myDiv', data, layout, {scrollZoom: true});

}

function viewEdges(edges_id,c) {

    var raw_vertices = document.getElementById(edges_id).value;
    console.log(raw_vertices);

    raw_vertices = raw_vertices.split(" ").filter(function (el) {
        return el.length != 0
    });
    console.log(raw_vertices);

    raw_vertices = raw_vertices.map(function (e) {
        e = parseFloat(Number(e));
        return e;
    });
    console.log(raw_vertices);

    var trace = {
        x: raw_vertices.filter((a, i) => i % 2 === 0),
        y: raw_vertices.filter((a, i) => i % 2 === 1),
        mode: 'markers',
        type: 'scatter'
    };

    for (i = 0; i < raw_vertices.length / 4; i++) {
        current_line = {
            type: 'line',
            x0: raw_vertices[i * 4],
            y0: raw_vertices[i * 4 + 1],
            x1: raw_vertices[i * 4 + 2],
            y1: raw_vertices[i * 4 + 3],
            line: {
                color: c,
                width: 3
            }
        };
        layout.shapes.push(current_line);
    }

    data.push(trace);
    
    Plotly.newPlot('myDiv', data, layout, {scrollZoom: true});

}
