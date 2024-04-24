#!/bin/bash	

ngc base-command job run \
	--name "interactive" \
	--priority "NORMAL" \
	--order "50" \
	--preempt "RUNONCE" \
	--min-timeslice "0s" \
	--total-runtime "0s" \
	--ace "servicenow-iad2-ace" \
	--org "e3cbr6awpnoq" \
	--instance "dgxa100.80g.1.norm" \
	--result "/result" \
	--image "e3cbr6awpnoq/research/adversarial_triggers:0.0.1" \
	--workspace "adversarial_triggers_iad2:/scratch:RW" \
	--port "8383" \
	--commandline "source setup.sh && sleep 3600"
