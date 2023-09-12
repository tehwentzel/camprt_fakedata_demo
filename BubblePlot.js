var OrganBubblePlot = (function(){
	"use strict";
	var instance;
	var num_organs;
	function Graph(){
		var self = this;
	}
	Graph.init = function(target, patientInternalId, patientData){
		div = document.getElementById(target);
		this.data = patientData;
		this.ids = patientData.getInternalIdList();
		this.setPatient(patientInternalId);
		this.width = div.clientWidth;
		this.height = div.clientHeight
		this.xMargin = .005*this.width;
		this.yMargin = .05*this.height;
		this.binWidth = (this.width - 2*this.xMargin)/num_organs;
		this.xAxisSize = 110;
		this.yAxisSize = 30;
		
		d3.selectAll('.bubbleSvg').remove();
		this.svg = d3.select("#"+target).append('svg')
						.attr('class', 'bubbleSvg')
						.attr('width', this.width)
						.attr('height', this.height)
						.append('g');
		this.tooltip = d3.select("div.bubbletooltip")
				.style('visibility','hidden');
		this.drawAxes();
		this.drawOrgans();
	}
	
	Graph.switchPatient = function(patientInternalId){
		this.setPatient(patientInternalId);
		this.drawOrgans();
	}
	
	Graph.setPatient = function(patientInternalId){
		this.patient = patientInternalId;
		this.organList = this.data.getOrganList(patientInternalId);
		num_organs = this.organList.length;
		
		this.patientMatches = [];
		var matches = this.data.getPatientMatches(patientInternalId);
		for(var i = 1; i < matches.length; i++){
			let match = matches[i];
			this.patientMatches.push( match );
		}
	}
	
	Graph.drawOrgans = function(){
		var [xAxis, yAxis] = this.getScales();
		var rAxis = this.getRScale(this.patient, this.binWidth);
		this.organList.forEach(function(d){
			this.drawOrganBubbles(d, xAxis, yAxis, rAxis);
		},this);
		this.drawDoseRects(xAxis, yAxis);	
	}
	
	Graph.drawAxes = function(){
		var[xScale, yScale] = this.getScales();
		var organList = this.organList;
		var self = this;
		var xAxis = d3.axisBottom(xScale)
			.ticks(this.organList.length)
			.tickFormat(function(d){
				return organList[d];
			});
		var yAxis = d3.axisLeft(yScale).ticks(5);
		var translate = "translate( 0," + (this.height - .5*this.yMargin- this.xAxisSize).toString() + ")";
		this.svg.append('g')
			.attr('class', 'axis')
			.attr('transform', translate)
			.call(xAxis)
			.selectAll('text')
			.attr('transform', 'rotate(-45)')
			.style('text-anchor', 'end')
		this.svg.append('g')
			.attr('class', 'axis')
			.attr('transform', 'translate(' + (.5*this.xMargin + this.yAxisSize) + ', ' + .5*this.yMargin + ')' )
			.call(yAxis);
			
		var tickSpaces = [];
		for(var i = 0; i < organList.length; i++){
			tickSpaces.push(i);
		}
		this.svg.selectAll('.axisLine')
			.data(tickSpaces).enter()
			.append('line')
			.attr('class', 'axisLine')
			.attr('id', function(d) {return organList[d] + 'axisLine';})
			.attr('x1', function(d){return xScale(d);})
			.attr('x2', function(d){return xScale(d);})
			.attr('y2', self.yMargin)
			.attr('y1', self.height - .5*self.yMargin- self.xAxisSize)
			.attr('stroke', 'silver')
			.attr('stroke-width', .05*self.binWidth)
		this.svg.selectAll('.bins')
			.data(tickSpaces).enter()
			.append('rect')
			.attr('class', '.bins')
			.attr('x', function(d){ return xScale(d) - self.binWidth/2;})
			.attr('y', self.yMargin)
			.attr('width', self.binWidth)
			.attr('height', self.height - self.yMargin - self.xAxisSize)
			.attr('opacity', 0)
			.on('mouseover', function(d){
				// var axisLine = d3.select( "#" + organList[d] + 'axisLine' )
				// axisLine.attr('stroke', 'white')
					// .attr('stroke-width', .08*self.binWidth);
				var id = self.patient;
				var organ = self.organList[d];
				Controller.brushOrgan(organ);
				self.tooltip.html( organ + '</br>'
				+ 'Predicted: ' + self.data.getEstimatedDose(id, organ) + ' Gy </br>'
				+ 'Actual: ' + self.data.getMeanDose(id, organ) + ' Gy')
				.style('left', d3.event.pageX + .5*self.binWidth + 'px')
				.style('top', d3.event.pageY - 50 + 'px');
				self.tooltip.transition().duration(50).style('visibility','visible');
				d3.select(this).on('mousemove', function(d){
					self.tooltip.style('left', d3.event.pageX + .5*self.binWidth + 'px')
						.style('top', d3.event.pageY - 50 + 'px');
				});
			}).on('mouseout', function(d){
				// var axisLine = d3.select( "#" + organList[d] + 'axisLine' )
				// axisLine.attr('stroke', 'silver')
					// .attr('stroke-width', .05*self.binWidth);
				//var id = self.patient;
				var organ = self.organList[d];
				Controller.unbrushOrgan();
				self.tooltip.transition().duration(50).style('visibility', 'hidden');
				d3.select(this).on('mousemove', null);
			});;
	}
	
	Graph.drawDoseRects = function(xScale, yScale){
		var patient = this.patient;
		var width = .8*this.binWidth;
		var height = .15*this.binWidth;
		var self = this;
		var getShape = d3.symbol().type(d3.symbolCross)
			.size((width**2)/4);
		d3.selectAll('.doseRect').remove();
		var drawDose = function(variable, color){
			var getY = function(o) {
				if(variable == 'meanDose'){
					var dose = self.data.getMeanDose(self.patient, o);
				} else{
					var dose = self.data.getEstimatedDose(self.patient, o);
				}
				return yScale(dose) - height/2; 
			};
			return self.svg.selectAll('.doseRect').filter('.'+variable)
				.data(self.organList).enter()
				.append('path')
				.attr('class', 'doseRect ' + variable)
				.attr('id', function(){ 
					return (variable=='meanDose')? ('organBubble' + self.patient): null;
				})
				.attr('d', getShape)
				.attr('transform', function(d,i){ 
					return 'translate(' + xScale(i) + ',' + getY(d) + ')';
				})
				.attr('fill', color)
				.attr('width', width)
				.attr('height', height)
				.attr('opacity', .9)
				.attr('stroke-width', 1)
				.attr('stroke', 'black');
		}
		drawDose('estimatedDose', '#3d32ff');
		drawDose('meanDose', this.data.getClusterColor(patient));
	}
	
	Graph.drawOrganBubbles = function(organName, xScale, yScale, rScale){
		var position = this.organList.indexOf(organName);
		var x = xScale(position);
		var self = this;
		var patient = this.patient;
		var tooltip = this.tooltip;
		var getRadius = function(x){
			var score = self.data.getSimilarity(self.patient, x);
			return rScale(score);
		}
		d3.selectAll('.organDot').filter('.' + organName).remove();
		var bubbles = this.svg.selectAll('.organDot').filter('.' + organName)
			.data(this.patientMatches).enter()
			.append('circle')
			.attr('class', 'organDot ' + organName)
			.attr('id', 
				function(d){ return ('organBubble'+d);} )//for selecting and brushing
			.attr('cx', x)
			.attr('cy', function(d){ 
				return yScale(self.data.getMeanDose(d, organName));})
			.attr('r', getRadius)
			.attr('opacity', .4)
			.attr('fill', function(d){ 
				return self.data.getClusterColor(d);
			})
			.attr('stroke', 'black')
			.attr('stroke-width', .1)
			.on('mouseover', function(d){
				tooltip.html(self.data.getPatientName(d) + '</br>'
				+ 'dose: ' + self.data.getMeanDose(d, organName).toFixed(2) + ' Gy')
				.style('left', d3.event.pageX + 10 + 'px')
				.style('top', d3.event.pageY - 30 + 'px');
				tooltip.transition().duration(50).style('visibility','visible');
				d3.select(this)
					.moveToFront()
					.transition().duration(50)
					.attr('opacity', 1)
					.attr('stroke-width', .3)
					.attr('stroke', 'white');
				Controller.brush(d);
			}).on('mouseout', function(d){
				tooltip.transition().duration(50).style('visibility', 'hidden');
				d3.select(this).transition().duration(50)
					.attr('opacity', .5)
					.attr('stroke', 'black')
					.attr('stroke-width', .1);
				Controller.unbrush(d);
			});
	}
	
	Graph.getRScale = function(patient, width){
		var ratio = .4;
		var similarityScores = this.data.getPatientSimilarityScores(patient);
		var scoreExtent = d3.extent( similarityScores.slice(1), function(d) {return d;});
		var rScale = d3.scalePow().exponent(2)
			.domain(scoreExtent)
			.range([ratio*width/2, width*(1-ratio)/2]);
		return(rScale)
	}
	
	Graph.getScales = function(){
		var self = this;
		var xScale = d3.scaleLinear()
			.domain([0, this.organList.length])
			.range([this.binWidth/2 + this.xMargin + this.yAxisSize, this.width - this.xMargin - this.binWidth/2]);
		//convoluted way of getting ths max and min dose for everything
		var allPatients = [...this.patientMatches];
		allPatients.push(this.patient);
		let maxDose = 0;
		let minDose = 0;
		allPatients.forEach(function(d){
			this.organList.forEach(function(organ){
				//console.log(d + ' ' + organ);
				var newDose = self.data.getMeanDose(d, organ);
				maxDose = (maxDose > newDose)? maxDose: newDose;
				minDose = (minDose < newDose)? minDose: newDose;
			});
		}, this);
		var yScale = d3.scaleLinear()
			.domain( [minDose, maxDose])
			.range([this.height - this.yMargin - this.xAxisSize, this.yMargin]);
		return [xScale, yScale];
	}
	return Graph;
}());