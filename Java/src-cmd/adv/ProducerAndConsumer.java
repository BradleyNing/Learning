public class ProducerAndConsumer {
	public static void main(String[] args) {
		Tickets t = new Tickets(6);
		new Consumer(t).start();
		new Producer(t).start();
	}
}

class Producer extends Thread {
	Tickets t = null;
	public Producer(Tickets t) { this.t = t;}

	public void run() {
		while (t.make_idx < t.size) {
			//synchronized(t) {
				System.out.println("Producer makes ticket "+ (++t.make_idx));
				t.available = true;
			//}
		}
		System.out.println("Producer ends!");
	}
}

class Consumer extends Thread {
	Tickets t=null;
	int i=0;
	public Consumer(Tickets t) { this.t = t;}

	public void run() {
		while (i < t.size) {
			//synchronized(t) {
				if(t.available==true && i<=t.make_idx)
					System.out.println("Consumer buys a ticket " + (++i));
				if (i==t.make_idx) {
					try{Thread.sleep(1);} catch(Exception e){;}
					t.available = false;
				}
			//}
			System.out.println("Consumer buying ended");
		}
	}
}

class Tickets {
	int size;
	int make_idx=0;
	int sell_idx=0;
	boolean available=false;
	public Tickets(int size) {this.size = size;}
}