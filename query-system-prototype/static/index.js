
function generateImageCard(image_id, header_text) {

    var image_card = document.createElement("div");
    image_card.classList.add("image-card");

    var image_card_header = document.createElement("div");
    image_card_header.classList.add("image-card-header");
    image_card_header.appendChild(document.createTextNode(header_text))

    image_card.appendChild(image_card_header);

    var image = new Image(256, 256);
    image.classList.add("image-card-image");
    image.src = `/download-image/${image_id}`;

    image_card.appendChild(image);

    return image_card;
}


class AI4HDRGraph {
    constructor(nodeLeftClickCallback = false) {
        this.m_ActiveThumbNails = [];

        this.m_GraphData = {
            nodes: [],
            links: []
        };

        this.m_NodeSet = new Set();
        this.m_LinkSet = new Set();

        this.m_MaxDataDistance = 3000;
        
        this.m_MaxLinkWidth = 20;
        this.m_LinkWidth = 85;

        var width = document.getElementById("canvas-view").getBoundingClientRect().width;
        var height = document.querySelector("#canvas-view").getBoundingClientRect().height;

        this.m_Graph = ForceGraph3D();
        this.m_Graph.width(width);
        this.m_Graph.height(height);

        this.m_Graph(document.getElementById('graph-3d'))
            .nodeLabel(node => node.id)
            .graphData(this.m_GraphData)
            .linkWidth('width')
            .cameraPosition({ z: 600 })
            .d3Force("link", d3.forceLink().distance(d => d.distance))
            .d3Force("charge", d3.forceManyBody().theta(1.0).strength(-50));

        if (nodeLeftClickCallback != false) {
            this.m_Graph.onNodeClick((node) => {
                var node_id = node.id;
                nodeLeftClickCallback(node, this);
            });
        }

        this.m_Graph.onNodeRightClick((node) => {
            this.removeNode(node.id);
        });

        // Initialize with random values
        this.clearAndRandomize();                
    }


    /********** PUBLIC METHODS **********/

    updateGraphData() {
        this.m_Graph.graphData(this.m_GraphData);
    }


    clearAndRandomize() {
        this.clear();

        $.get(`/get-random/${5}`, (data) => {

            for (var i = 0; i < data.length; i++) {
                this.addNode(data[i].idx);
                this.addNodeNearestNeighbors(data[i].idx);
            }
            
            this.updateActiveThumbNails(data);
        });

    }


    clear() {
        this.m_NodeSet.clear();
        this.m_LinkSet.clear();
        this.m_GraphData.nodes.splice(0, this.m_GraphData.nodes.length);
        this.m_GraphData.links.splice(0, this.m_GraphData.links.length);
        this.m_Graph.graphData(this.m_GraphData);
    }


    removeNode(node_id) {
        var index = this.m_GraphData.nodes.findIndex((node) => {
            return node.id == node_id
        });
        this.m_GraphData.links = this.m_GraphData.links.filter(l => l.source.id != node_id && l.target.id != node_id); // Remove links attached to node
        
        this.m_NodeSet.delete(node_id);
        this.m_GraphData.nodes.splice(index, 1); // Remove node
        this.m_Graph.graphData(this.m_GraphData);

        for (var i = 0; i < this.m_ActiveThumbNails.length; i++) {
            if (this.m_ActiveThumbNails[i]["node"].id == node_id) {
                this.m_Graph.scene().remove(this.m_ActiveThumbNails[i]["sprite"]);
                break;
            }
        }
    }


    addNodeNearestNeighbors(query_id) {
        $.get(`/get-nearest-neighbors/${query_id}/${10}`, (data) => {
            var nearest_neighbors = data[0];

            console.log(`Querying image: id=${query_id}`)

            for (var i = 0; i < nearest_neighbors.length; i++) {
                this.addNode(nearest_neighbors[i]["idx"]);
                this.addEdge(query_id, 
                             nearest_neighbors[i]["idx"],
                             this.getConnectionWidth(nearest_neighbors, nearest_neighbors[i]["distance"]),
                             this.getConnectionDistance(nearest_neighbors, nearest_neighbors[i]["distance"]));
            }
            this.m_Graph.graphData(this.m_GraphData);
        });
    }


    addEdge(source, destination, edge_width=5, edge_length=85) {
        if (source != destination) {
            var edge = {
                source: source,
                target: destination,
                width: edge_width,
                distance: edge_length
            };

            if (!this.m_LinkSet.has(JSON.stringify({source: source, destination: destination}))) {
                this.m_GraphData.links.push(edge);
                this.m_LinkSet.add(JSON.stringify({source: source, destination: destination}));
            }
        }
    }
    

    addNode(node_id) {
        if (!this.m_NodeSet.has(node_id)) {
            this.m_GraphData.nodes.push({id: node_id});
            this.m_NodeSet.add(node_id);
        }
    }


    updateActiveThumbNails(new_thumbnail_ids) {
    
        for (var i = 0; i < this.m_ActiveThumbNails.length; i++) {
            this.m_Graph.scene().remove(this.m_ActiveThumbNails[i]["sprite"]);
        }

        this.m_ActiveThumbNails.splice(0, this.m_ActiveThumbNails.length);

        const { nodes, links } = this.m_Graph.graphData();

        for (var i = 0; i < new_thumbnail_ids.length; i++) {
            var map = new THREE.TextureLoader().load( `/download-image/${new_thumbnail_ids[i].id}` );
            map.minFilter = THREE.LinearFilter;
            var material = new THREE.SpriteMaterial( { map: map } );
            var sprite =  new THREE.Sprite( material );
            sprite.scale.set(75,75,5);
            
            var node;

            for (var j = 0; j < nodes.length; j++) {
                if (nodes[j].id == new_thumbnail_ids[i].idx) {
                    node = nodes[j];
                    break;
                }
            }

            sprite.position.set(node.x, node.y, node.z)
            this.m_Graph.scene().add(sprite);

            this.m_ActiveThumbNails.push({ "sprite": sprite, "node": node });
        }

        
    }

    updateThumbnailPositions() {
    
        for (var i = 0; i < this.m_ActiveThumbNails.length; i++) {
            var sprite = this.m_ActiveThumbNails[i]["sprite"];
            var node = this.m_ActiveThumbNails[i]["node"];
            sprite.position.set(node.x, node.y, node.z);
        }
    }


    getConnectionWidth(connections_list, distance) {
        var scale = 1.5;
        var all_distances = [];
        for (var i = 0; i < connections_list.length; i++) {
            all_distances.push(connections_list[i]["distance"]);
        }
        all_distances.sort();

        var size = connections_list.length;
        for (var i = 0; i < all_distances.length; i++) {
            if (all_distances[i] == distance) {
                return size * scale;
            }
            size--;
        }

        return 0;            
    }


    getConnectionDistance(connections_list, distance) {
        return 175;        
    }

}

function nodeLeftClickCallback(node, graph) {

    $.get(`/get-nearest-neighbors/${node.id}/${10}`, (data) => {
        var nearest_neighbors = data[0];

        var new_thumbnails = [];

        var image_container = document.getElementById("image-view");
        image_container.innerHTML = "";

        var query_image_card = generateImageCard(nearest_neighbors[0]["id"], "Query Image");
        image_container.appendChild(query_image_card);

        for (var i = 0; i < nearest_neighbors.length; i++) {

            if (i != 0) {
                
                query_image_card = generateImageCard(nearest_neighbors[i]["id"], `Distance: ${nearest_neighbors[i]["distance"]}`);
                image_container.appendChild(query_image_card);
            }
            
            graph.addNode(nearest_neighbors[i]["idx"]);
            graph.addEdge(node.id, 
                          nearest_neighbors[i]["idx"],
                          graph.getConnectionWidth(nearest_neighbors, nearest_neighbors[i]["distance"]),
                          graph.getConnectionDistance(nearest_neighbors, nearest_neighbors[i]["distance"]));
            new_thumbnails.push({"id": nearest_neighbors[i]["id"], "idx": nearest_neighbors[i]["idx"]});
        }
        graph.updateGraphData();
        graph.updateActiveThumbNails(new_thumbnails);
        
    });
}


window.onload = function() {

    var ai4hdr_graph = new AI4HDRGraph(nodeLeftClickCallback = nodeLeftClickCallback);

    setInterval(() => {
        ai4hdr_graph.updateThumbnailPositions();
    }, 10);
}