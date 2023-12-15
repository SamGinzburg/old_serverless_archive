let fs = require("fs");
let path = require("path");
var AWS = require('aws-sdk');
var s3 = new AWS.S3();
let spawnSync = require('child_process').spawnSync;


function bench(cb) {

    var regex1 = /interrupts\(.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\nLOC:\ +(.*?)\ /;
    var regex2 = /interrupts\(.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\nLOC:\ +\d+\ +(.*?)\ /;
    
    var regex3 = /interrupts2\(.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\nLOC:\ +(.*?)\ /;
    var regex4 = /interrupts2\(.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\n.+.+\n.+.\n.+\nLOC:\ +\d+\ +(.*?)\ /;

    var resp = "";
    var result = spawnSync('cat', ['/proc/interrupts'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    resp = "interrupts(" + String(savedOutput)+")interrupts\n";
    var results1 = resp.match(regex1);
    var results2 = resp.match(regex2);

    console.log("interrupts1l1("+results1[1]+")interrupts\n");
    console.log("interrupts1l2("+results2[1]+")interrupts\n");

    var savedOutput = result.stderr;
    console.log("stderr:\t" + String(savedOutput) + "\n");


    var result = spawnSync('/tmp/vmstat', ['-s'] ,{
   // var result = spawnSync('ls', ['/usr/bin/'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    console.log("instanceid(" + String(savedOutput)+")instanceid\n");

    resp = "instanceid(" + String(savedOutput)+")instanceid\n";


    var savedOutput = result.stderr;
    console.log("stderr:\t" + String(savedOutput) + "\n");

    console.time('benchmark');
    
    var result = spawnSync('/tmp/nqueens', [''] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });

    var savedOutput = result.stderr;
    console.log("stderr:\t" + String(savedOutput) + "\n");
    console.timeEnd('benchmark');

    var result = spawnSync('vmstat', ['-s'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    console.log("instanceid2(" + String(savedOutput)+")instanceid2\n");
    resp = "instanceid2(" + String(savedOutput)+")instanceid2\n";


    var savedOutput = result.stderr;
    console.log("stderr:\t" + String(savedOutput) + "\n");

    var result = spawnSync('cat', ['/proc/interrupts'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    resp = "interrupts2(" + String(savedOutput)+")interrupts2\n";
    var results3 = resp.match(regex3);
    var results4 = resp.match(regex4);
    
    console.log("interrupts2l1("+results3[1]+")interrupts\n");
    console.log("interrupts2l2("+results4[1]+")interrupts\n");

    var savedOutput = result.stderr;
    console.log("stderr:\t" + String(savedOutput) + "\n");

    var result = spawnSync('cat', ['/proc/stat'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    console.log(String(savedOutput));
    console.log(result.status);

    cb(null, resp);
}

exports.handler = function(event, context, callback) {
    console.log("event.probe:\t" + String(event.probe));

    process.env.PATH = process.env.PATH + ':' + process.env['LAMBDA_TASK_ROOT'] 
    //console.log(process.env);

    var result = spawnSync('cp', ['nqueens', '/tmp/nqueens'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    //console.log(String(savedOutput));

    var result = spawnSync('chmod', ['755', '/tmp/nqueens'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    //console.log(String(savedOutput));
    //console.log(result.status);


   var result = spawnSync('cp', ['vmstat', '/tmp/vmstat'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    //console.log(String(savedOutput));

    var result = spawnSync('chmod', ['755', '/tmp/vmstat'] ,{
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    //console.log(String(savedOutput));
    //console.log(result.status);

    if (event.probe === 'true') {
    
        bench(callback);

    } else {
    }
};
