public class ThreadRunnableTest {
	public static void main(String[] args) {
		System.out.println("main thread/ThreadTest started");
		FactorialRunnable thread = new FactorialRunnable(10);
		new Thread(thread).start();
		try {
			Thread.sleep(1);
		} catch (Exception e) {
			System.out.println(e);
		}

		System.out.println("main thread started FactorialThread, sleeped 1ms and self-ended");
	}
}

class FactorialRunnable implements Runnable {
	private int num;

	public FactorialRunnable(int num) {
		this.num = num;
	}

	public void run() {
		int i = num;
		int result = 1;
		System.out.println("FactorialThread started");
		while (i > 0) {
			result = result*i;
			i--;
		}
		System.out.println("The factorial of " + num + " is " + result);
		System.out.println("FactorialThread ended");
	}
}