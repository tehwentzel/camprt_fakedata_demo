//function to move svg elements  to the front
//from https://github.com/wbkd/d3-extended
d3.selection.prototype.moveToFront = function() {  
  return this.each(function(){
	this.parentNode.appendChild(this);
  });
};
d3.selection.prototype.moveToBack = function() {  
  return this.each(function(){
	this.parentNode.insertBefore(this, this.parentNode.firstChild);
  });
};

function DoseScatterPlot(data){
	this.data = data;
	this.ids = data.getInternalIdList();
	this.selectedColor = '#e41a1c';
	this.getColor = function(d){ return data.getClusterColor(d)};
}

DoseScatterPlot.prototype.draw = function(target, selectedPerson = null){
	div = document.getElementById(target);
	this.width = div.clientWidth;
	this.height = div.clientHeight;
	this.xMargin = .04*this.width;
	this.yMargin = .03*this.width;
	this.axisLabelSize = 16;//pix
	this.clusterMargin = 20;
	this.baseOpacity = 0.75;
	this.highlightedOpacity = .95;
	this.clusterOpacity = .5;
	d3.select("#"+target).selectAll('.scatterSvg').remove();
	this.svg = d3.select("#"+target).insert('svg',':first-child')
					.attr('class', 'scatterSvg')
					.attr('width', this.width)
					.attr('height', this.height)
					.append('g');
	this.tooltip = d3.select("div.tooltip")
			.attr('class','tooltip')
			.style('visibility','hidden');
	this.switchAxisFunction('distance');
	this.drawCircles(selectedPerson);
	if (selectedPatient != null){
		this.highlightSelectedPatients(selectedPerson);
	}
	this.drawClusterCircles(this.clusterMargin);
	this.setupTooltip(selectedPerson);
	this.setupSwitchButtons();
	this.drawAxisLabels('Tumor-Organ Distance TSNE 1', 'Tumor-Organ Distance TSNE 2');
}

DoseScatterPlot.prototype.setupSwitchButtons = function(){
	var setSelected = function(element){
		element.style.opacity = 1;
	}
	var setUnselected = function(element){
		element.style.opacity = .4;
	}
	var doseButton = document.getElementById('doseScatterButton');
	var distanceButton = document.getElementById('distanceScatterButton');
	var stagingButton = document.getElementById('stagingScatterButton');
	var similarityButton = document.getElementById('similarityScatterButton');
	allButtons = [doseButton, distanceButton, stagingButton, similarityButton];
	var self = this;
	var onScatterButtonClick = function(btn, axisVariable){
		if(btn.style.opacity ==1)
			return 
		self.switchAxisVariable(axisVariable);
		allButtons.forEach(function(b){
			if(b != btn){
				setUnselected(b);
			}
		});
		setSelected(btn);
	};
	setSelected(distanceButton);
	setUnselected(doseButton);
	setUnselected(stagingButton);
	setUnselected(similarityButton);
	doseButton.addEventListener('click', function(){
		onScatterButtonClick(this, 'dose');
	});
	distanceButton.addEventListener('click', function(){
		onScatterButtonClick(this, 'distance');
	});
	stagingButton.addEventListener('click', function(){
		onScatterButtonClick(this, 'staging');
	});
	similarityButton.addEventListener('click', function(){
		onScatterButtonClick(this, 'similarity');
	});
}

DoseScatterPlot.prototype.drawAxisLabels = function(xLabel, yLabel, padding = 5){
	this.svg.selectAll('text').filter('.axisLabel').remove();
	var labelGroup = this.svg.append('g').attr('class', 'axisLabel');
	labelGroup.append('text').attr('class','axisLabel')
		.attr('id','scatterPlotXLabel')
		.attr('text-anchor', 'middle')
		.attr('font-size', this.axisLabelSize + 'px')
		.attr('x', .5*this.width)
		.attr('y', this.height - padding)
		.text(xLabel);
	labelGroup.append('text').attr('class','axisLabel')
		.attr('id','scatterPlotYLabel')
		.attr('text-anchor', 'middle')
		.attr('font-size', this.axisLabelSize + 'px')
		.attr('x', this.width - padding)
		.attr('y', .5*this.height)
		.text(yLabel)
		.attr('transform', 'rotate(-90,' +  (this.width - padding) + ',' + (.5*this.height) + ')');
}

DoseScatterPlot.prototype.renameAxis = function(xName = null, yName = null){
	if(xName != null){ d3.select('#scatterPlotXLabel').text(xName); }
	if(yName != null){ d3.select('#scatterPlotYLabel').text(yName); }
}

DoseScatterPlot.prototype.getAxisScales = function(){
	self = this;
	var xDomain = d3.extent(this.ids, function(d){return self.getXAxis(d);});
	var yDomain = d3.extent(this.ids, function(d){return self.getYAxis(d);})
	
	var xScale = d3.scaleLinear()
		.domain(xDomain)
		.range([this.xMargin, this.width - this.xMargin - 1.1*this.axisLabelSize]);
	var yScale = d3.scaleLinear()
		.domain(yDomain)
		.range([this.height - this.yMargin - 1.1*this.axisLabelSize, this.yMargin]);
	return([xScale, yScale])
}

DoseScatterPlot.prototype.getFeatureScales = function(){
	var self = this;
	var sizeScale = d3.scalePow().exponent(2)
		.domain( d3.extent(this.ids, function(d){ return data.getPatientMeanError(d); }) )
		.range([20, 250]);
	return sizeScale;
}


DoseScatterPlot.prototype.getShape = function(sizeScale){
	var self = this;
	return d3.symbol().type(function(d){
		if(d == selectedPatient){
			return d3.symbolCross;
		}
		let shape = (data.hasToxicity(d) == 1)? d3.symbolDiamond: d3.symbolCircle;
		return shape
	}).size(function(d){
		return sizeScale(self.data.getPatientMeanError(d));
	});
}

DoseScatterPlot.prototype.drawCircles = function(selectedPatient){
	var self = this;
	var [xScale, yScale] = this.getAxisScales();
	var sizeScale = this.getFeatureScales();
	var getColor = this.getColor;
	
	var getShape = self.getShape(sizeScale) 
	
	var triangle = d3.symbol().type(d3.symbolCross);
	d3.selectAll('.point').remove();
	this.circles = this.svg.selectAll('.point')
		.data(this.ids).enter()
		.append('path')
		.attr('class', 'point')
		.attr('id', function(d){ return 'scatterDot' + d;})
		.attr('d', getShape)
		.attr('transform', function(d){
			return 'translate(' + xScale(self.getXAxis(d)) + ',' + yScale(self.getYAxis(d)) + ')';
		});
	this.circles
		.attr('fill', function(d){ 
			return getColor(d);})
		.attr('stroke', 'black')
		.attr('opacity', self.highlightedOpacity)
		.attr('stroke-width', 1);
	this.circles.on('click', function(d){
		switchPatient(d);//from camprt.js
	});
}

DoseScatterPlot.prototype.drawClusterCircles = function(margin){
	var self = this;
	var clusters = new Map();
	var clusterColors = new Map();
	var [xScale, yScale] = this.getAxisScales();
	var toPoint = function(d){
		var x = xScale(self.getXAxis(d));
		var y = yScale(self.getYAxis(d));
		return [x,y];
	}
	this.ids.forEach(function(d){
		var cluster = self.data.getCluster(d);
		if(!clusters.has(cluster)){
			clusters.set(cluster, []);
			var color = self.getColor(d);
			clusterColors.set(cluster, color);
		}
		var current = clusters.get(cluster);
		current.push(toPoint(d))
		clusters.set(cluster, current);
	}, clusters);
	var interpolateLine = function(x0, x1){
		var magnitude = Math.sqrt((x1[1] - x0[1])**2 + (x1[0] - x0[0])**2);
		var vect = [ (x1[0] - x0[0])/magnitude, (x1[1] - x0[1])/magnitude ];
		var point = [x1[0] + vect[0]*margin, x1[1] + vect[1]*margin];
		return point;
	}
	var offsetHulls = [];
	for (var [key, value] of clusters.entries()) {
		var hull = [];
		if(value.length <= 1){
			continue;
		}
		try{
			var convexHull = d3.polygonHull(value);
			var centroid = d3.polygonCentroid(convexHull);
		}catch{
			var convexHull = value;
			var centroid = [0,0];
			value.forEach(function(point){
				centroid[0] += point[0];
				centroid[1] += point[1];
			});
			centroid[0] /= value.length;
			centroid[1] /= value.length;
			convexHull.push([centroid[0] + margin, centroid[1] + margin]);
			convexHull.splice(1, 0, [centroid[0] - margin, centroid[1] - margin]);
		}
		convexHull.forEach(function(point){
			var offsetPoint = interpolateLine(centroid, point);
			hull.push(offsetPoint);
		});
		hull.color = clusterColors.get(key);
		hull.cluster = +key;
		offsetHulls.push(hull);
	}
	var arcPath = d3.line()
		.x(function(d){ return d[0];})
		.y(function(d){ return d[1];})
		.curve(d3.curveCardinalClosed);
	var arc = this.svg.selectAll('.clusterCurves')
		.data(offsetHulls);
	arc.exit().remove();
	arc.enter()
		.append('path')
		.attr('class','clusterCurves')
		.attr('fill', 'none')
		.attr('stroke-width', margin/3)
		.attr('opacity',self.clusterOpacity)
		.merge(arc).transition().duration(800)
		.attr('d', function(d){return arcPath(d);})
		.attr('stroke', function(d) {return d.color;});
	d3.selectAll('.clusterCurves').moveToBack();
	this.setupCurveTooltip();
}

DoseScatterPlot.prototype.setupCurveTooltip = function(){
	var clusterStats = new Map();
	var self = this;
	this.ids.forEach(function(d){
		var cluster = self.data.getCluster(d);
		if(!clusterStats.has(cluster)){
			var base = new Object();
			base.numPoints = 0;
			base.meanDose = 0;
			base.meanError = 0;
			clusterStats.set(cluster, base);
		}
		var current = clusterStats.get(cluster);
		current.numPoints += 1;
		current.meanError += self.data.getPatientMeanError(d);
		current.meanDose += self.data.getPatientMeanDose(d);
		clusterStats.set(cluster, current);
	});
	for(var stats of clusterStats.values()){
		stats.meanError = stats.meanError/stats.numPoints;
		stats.meanDose = stats.meanDose/stats.numPoints;
	}
	d3.selectAll('path').filter('.clusterCurves')
		.on('mouseover', function(d){
			d3.select(this).attr('opacity', 1);
			var stats = clusterStats.get(d.cluster);
			self.tooltip.html('Cluster ' + d.cluster + '</br>'
			+ 'Size: ' + stats.numPoints + '</br>'
			+ 'Mean Dose: ' + stats.meanDose.toFixed(1) + 'Gy </br>'
			+ 'Mean Prediction Error: ' + stats.meanError.toFixed(1) + '%' )
				.style('left', d3.event.pageX + 8 + 'px')
				.style('top', d3.event.pageY - 20 + 'px');
			self.tooltip.transition().duration(50).style('visibility','visible');
			let color = data.getClusterColor(d.cluster, patient = false);
			d3.selectAll('.point[fill=' + '\"' + color.toString() + '\"' + "][opacity=" + '\"' + self.baseOpacity + '\"' + "]")
				.attr('opacity', self.highlightedOpacity);
		}).on('mouseout', function(d){
			d3.select(this).attr('opacity', self.clusterOpacity);
			self.tooltip.transition().duration(50).style('visibility', 'hidden');
			let color = data.getClusterColor(d.cluster, patient=false);
			d3.selectAll('.point[fill=' + '\"' + color + '\"' + "][opacity=" + '\"' + self.highlightedOpacity + '\"' + "]")
				.attr('opacity', self.baseOpacity);
		});
}

DoseScatterPlot.prototype.setAxisVariable = function(axisFunction, axis){
	axis = +axis;
	if(axis != 1 || axis != 0){
		console.log('invalid axis to scatterplot setAxisVariable.  Value mut be 1 or 0');
	}
	if(axis == 1){
		this.getXAxis = axisFunction;
	}else{
		this.getYAxis = axisFunction;
	}
}

DoseScatterPlot.prototype.animateAxisChange = function(){
	var [xScale, yScale] = this.getAxisScales();
	this.circles.transition().duration(800)
		.attr('transform', function(d){
			return 'translate(' + xScale(self.getXAxis(d)) + ',' + yScale(self.getYAxis(d)) + ')';
		});
	this.drawClusterCircles(this.clusterMargin);
}
DoseScatterPlot.prototype.switchAxisFunction = function(type){
	if(type == 'distance'){
		this.getXAxis = function(d){ 
			var pca1 = self.data.getDistancePCA(d, 1);
			return  pca1;//Math.sign(pca1)*Math.pow(Math.abs(pca1),.4); 
		};
		this.getYAxis = function(d){ 
			var pca2 = self.data.getDistancePCA(d, 2);
			return pca2;//Math.sign(pca2)*Math.pow(Math.abs(pca2),.2); 
		};
	} else if(type == 'staging'){
		this.getXAxis = function(d){ 
			var volume = self.data.gtvpVol(d);
			return (volume > Math.E)? Math.log(volume): volume; 
		};
		this.getYAxis = function(d){ 
			var volume = self.data.gtvnVol(d);
			return (volume > Math.E)? Math.log(volume): volume; 
		};
	} else if(type == 'similarity'){
		this.getXAxis = function(d){
			return self.data.getSimilaritySpaceEmbedding(d, 1);
		};
		this.getYAxis = function(d){
			return self.data.getSimilaritySpaceEmbedding(d, 2);
		};
	} else{
		this.getXAxis = function(d){ return self.data.getDosePCA(d, 1); };
		this.getYAxis = function(d){ return self.data.getDosePCA(d, 2); };
	}
}

DoseScatterPlot.prototype.switchAxisVariable = function(type){
	this.switchAxisFunction(type);
	if(type == 'distance'){
		this.renameAxis('Tumor-Organ Distance TSNE 1', 'Tumor-Organ Distance TNSE 2');
	} else if(type == 'staging'){
		this.renameAxis('GTVp Volume', 'GTVn Volume');
	} else if(type == 'similarity'){
		this.renameAxis('2D Similarity Based Embedding', '');
	}
	else{
		this.renameAxis('Dose PC 1', 'Dose PC 2');
	}
	this.animateAxisChange();
}


DoseScatterPlot.prototype.highlightSelectedPatients = function(selectedPerson){
	//recolor everything first
	var getColor = this.getColor;
	var self = this;
	
	var sizeScale = this.getFeatureScales();
	
	var getShape = self.getShape(sizeScale)
		
	this.circles
		.attr('d', getShape)
		.attr('fill', function(d){ 
			return getColor(d);})
		.attr('stroke', 'black')
		.attr('stroke-width', 1)
		.attr('opacity', self.baseOpacity);
	//get entries linked to the person
	var selectedMatches = new Array();
	self.data.getPatientMatches(selectedPerson).forEach(function(id){
		selectedMatches.push(id);
	}, this);
	//recolor people matched with selected patient, make them darker and colorfuler
	selectedMatches.forEach(function(x){
		d3.select('#scatterDot' + x)
			.attr('opacity', 1)
			.attr('stroke-width', 2)
			.moveToFront();
	});
	//make main patient red and stuff
	d3.select('#scatterDot' + selectedPerson)
		.attr('opacity', 1)
		.attr('stroke-width', 2)
		.moveToFront();
}

DoseScatterPlot.prototype.setupTooltip = function(selectedPatient){
	var tooltip = this.tooltip;
	var self = this;

	this.circles.on('mouseover', function(id){
		tooltip.html(self.data.getPatientName(id) + '</br>' 
			+ 'Dose: ' + self.data.getPatientMeanDose(id).toFixed(3) + ' Gy</br>'
			+ 'Error: ' + self.data.getPatientMeanError(id).toFixed(3) + ' %</br>'
			+ 'Cluster: ' + self.data.getCluster(id) + '</br>'
			+ 'x Value: ' + self.getXAxis(id).toFixed(3) + '</br>'
			+ 'y Value: ' + self.getYAxis(id).toFixed(3))
			.style('left', d3.event.pageX + 10 + 'px')
			.style('top', d3.event.pageY - 30 + 'px');
		tooltip.transition().duration(50).style('visibility','visible');
		Controller.brush(id);
	}).on('mouseout', function(d){
		tooltip.transition().duration(50).style('visibility', 'hidden');
		Controller.unbrush(d);
	});
}